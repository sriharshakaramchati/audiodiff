#!/usr/bin/env python3
"""
audiodiff: a perceptual diff for two versions of an audio file.

Built for layer-by-layer AI music workflows: you generated a patch,
spliced it in, and want to see (and hear) exactly what changed.

Usage:
    python audiodiff.py old.wav new.wav -o report.html
    python audiodiff.py old.wav new.wav --align -o report.html

Output is a single self-contained HTML report:
  - both waveforms rendered as DAW-style lanes
  - a diff lane showing per-window spectral distance
  - a list of changed regions with timestamps and severity
  - embedded audio players; click any region to A/B it

Dependencies: numpy, scipy, soundfile. ffmpeg (optional) for
compact audio embedding, otherwise WAV is embedded.
"""

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.signal as sps
import soundfile as sf

# ---------------------------------------------------------------- loading

TARGET_SR = 22050  # analysis rate; plenty for spectral comparison


def load_mono(path, target_sr=TARGET_SR):
    """Load any audio file as mono float32 at target_sr."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        mono = sps.resample_poly(mono, target_sr // g, sr // g)
    return mono.astype(np.float32), target_sr


def align_offset(a, b, sr, max_shift_s=5.0):
    """Estimate the lag of b relative to a via onset-envelope
    cross-correlation. Returns lag in samples (positive: b is late)."""
    hop = 512
    ea = onset_envelope(a, hop)
    eb = onset_envelope(b, hop)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    corr = sps.correlate(eb, ea, mode="full")
    lags = sps.correlation_lags(len(eb), len(ea), mode="full")
    max_lag = int(max_shift_s * sr / hop)
    mask = np.abs(lags) <= max_lag
    lag_frames = lags[mask][np.argmax(corr[mask])]
    return int(lag_frames * hop)


def onset_envelope(x, hop):
    frames = frame_signal(x, 1024, hop)
    energy = (frames ** 2).mean(axis=1)
    d = np.diff(energy, prepend=energy[:1])
    return np.maximum(d, 0.0)


def frame_signal(x, win, hop):
    n = 1 + max(0, (len(x) - win)) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]

# ---------------------------------------------------------------- analysis

N_FFT = 2048
HOP = 512
N_BANDS = 48


def log_band_energies(x, sr):
    """Log-spaced band energies per STFT frame -> (frames, N_BANDS)."""
    f, t, Z = sps.stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                       padded=False, boundary=None)
    mag = np.abs(Z)  # (freqs, frames)
    lo, hi = 40.0, sr / 2 * 0.95
    edges = np.geomspace(lo, hi, N_BANDS + 1)
    bands = np.zeros((mag.shape[1], N_BANDS), dtype=np.float32)
    for i in range(N_BANDS):
        sel = (f >= edges[i]) & (f < edges[i + 1])
        if sel.any():
            bands[:, i] = mag[sel].mean(axis=0)
    return np.log1p(bands * 100.0), t


def spectral_distance(A, B):
    """Per-frame distance between two band-energy matrices.
    Combines normalized L2 (level changes) and cosine (timbre changes)."""
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    l2 = np.linalg.norm(A - B, axis=1)
    scale = np.linalg.norm(A, axis=1) + np.linalg.norm(B, axis=1) + 1e-6
    l2n = l2 / scale
    dot = (A * B).sum(axis=1)
    cos = dot / ((np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + 1e-6)
    cosd = 1.0 - np.clip(cos, -1.0, 1.0)
    d = 0.6 * l2n + 0.4 * cosd
    # smooth ~0.25s
    k = max(1, int(0.25 * TARGET_SR / HOP))
    kernel = np.hanning(2 * k + 1)
    kernel /= kernel.sum()
    return np.convolve(d, kernel, mode="same")


def find_regions(dist, times, min_dur=0.35, gap=0.9):
    """Threshold the distance curve into contiguous changed regions."""
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med))) + 1e-6
    thr = med + 4.0 * mad
    floor = 0.06  # ignore numeric dust on near-identical files
    thr = max(thr, floor)
    hot = dist > thr

    regions = []
    start = None
    for i, h in enumerate(hot):
        if h and start is None:
            start = i
        elif not h and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(hot)))

    # merge nearby, drop tiny
    merged = []
    for s, e in regions:
        if merged and times[s] - times[merged[-1][1] - 1] < gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        t0, t1 = float(times[s]), float(times[min(e, len(times) - 1)])
        if t1 - t0 >= min_dur:
            seg = dist[s:e]
            out.append({
                "start": round(t0, 2),
                "end": round(t1, 2),
                "peak": round(float(seg.max()), 3),
                "mean": round(float(seg.mean()), 3),
            })
    return out, thr

# ---------------------------------------------------------------- rendering

WAVE_POINTS = 1600


def peak_envelope(x, points=WAVE_POINTS):
    n = len(x)
    if n == 0:
        return np.zeros(points)
    step = max(1, n // points)
    trimmed = x[: (n // step) * step]
    peaks = np.abs(trimmed.reshape(-1, step)).max(axis=1)
    if len(peaks) < points:
        peaks = np.pad(peaks, (0, points - len(peaks)))
    return peaks[:points]


def encode_audio(path, prefer_ffmpeg=True):
    """Return (mime, base64) for embedding. Uses ffmpeg -> ogg if present."""
    if prefer_ffmpeg and shutil.which("ffmpeg"):
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "2",
                 "-c:a", "libvorbis", "-q:a", "3", tmp_path],
                capture_output=True)
            if r.returncode == 0:
                with open(tmp_path, "rb") as fh:
                    return "audio/ogg", base64.b64encode(fh.read()).decode()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    # fallback: re-encode to 16-bit wav at 32k for size sanity
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, data, sr, subtype="PCM_16", format="WAV")
    return "audio/wav", base64.b64encode(buf.getvalue()).decode()


def fmt_time(t):
    m, s = divmod(t, 60.0)
    return f"{int(m)}:{s:05.2f}"


def build_html(name_a, name_b, dur, env_a, env_b, dist_norm, regions,
               audio_a, audio_b, offset_s):
    payload = {
        "nameA": name_a, "nameB": name_b, "dur": round(dur, 3),
        "envA": [round(float(v), 4) for v in env_a],
        "envB": [round(float(v), 4) for v in env_b],
        "dist": [round(float(v), 4) for v in dist_norm],
        "regions": regions,
        "mimeA": audio_a[0], "b64A": audio_a[1],
        "mimeB": audio_b[0], "b64B": audio_b[1],
        "offset": round(offset_s, 3),
    }
    template = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    return template


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>audiodiff report</title>
<style>
  :root{
    --bg:#101318; --panel:#171b22; --line:#232935;
    --txt:#c8cdd8; --dim:#6b7484;
    --wave-a:#5b8dbe; --wave-b:#4fae8d;
    --hot:#e0603a; --hot-soft:rgba(224,96,58,.16);
    --mono:'SF Mono',ui-monospace,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--txt);font-family:var(--mono);
       font-size:13px;line-height:1.5;padding:32px 28px 64px}
  header{display:flex;justify-content:space-between;align-items:baseline;
         border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:22px}
  h1{font-size:15px;font-weight:600;letter-spacing:.04em}
  h1 span{color:var(--hot)}
  .meta{color:var(--dim);font-size:11px;text-align:right}
  .lane{background:var(--panel);border:1px solid var(--line);border-radius:6px;
        margin-bottom:10px;position:relative;overflow:hidden}
  .lane-label{position:absolute;top:6px;left:10px;font-size:10px;
              color:var(--dim);letter-spacing:.08em;text-transform:uppercase;
              z-index:3;pointer-events:none}
  .lane-label b{color:var(--txt);font-weight:600}
  canvas{display:block;width:100%}
  #cursor{position:absolute;top:0;bottom:0;width:1px;background:#fff;
          opacity:.7;pointer-events:none;z-index:4;display:none}
  #timeline{position:relative}
  .controls{display:flex;gap:8px;align-items:center;margin:16px 0 26px;flex-wrap:wrap}
  button{background:var(--panel);border:1px solid var(--line);color:var(--txt);
         font-family:var(--mono);font-size:12px;padding:7px 14px;border-radius:5px;
         cursor:pointer}
  button:hover{border-color:var(--dim)}
  button:focus-visible{outline:2px solid var(--wave-a);outline-offset:2px}
  button.active{border-color:var(--hot);color:var(--hot)}
  .hint{color:var(--dim);font-size:11px;margin-left:auto}
  table{width:100%;border-collapse:collapse;margin-top:10px}
  th{color:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:.1em;
     text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
  td{padding:9px 10px;border-bottom:1px solid var(--line);font-size:12px}
  tr.region{cursor:pointer}
  tr.region:hover td{background:var(--hot-soft)}
  .sev{display:inline-block;height:8px;border-radius:2px;background:var(--hot);
       vertical-align:middle}
  .none{color:var(--dim);padding:20px 10px}
  h2{font-size:11px;color:var(--dim);text-transform:uppercase;
     letter-spacing:.12em;margin:28px 0 4px}
</style>
</head>
<body>
<header>
  <h1>audio<span>diff</span></h1>
  <div class="meta" id="meta"></div>
</header>

<div id="timeline">
  <div class="lane"><div class="lane-label">A · <b id="labA"></b></div>
    <canvas id="wA" height="110"></canvas></div>
  <div class="lane"><div class="lane-label">B · <b id="labB"></b></div>
    <canvas id="wB" height="110"></canvas></div>
  <div class="lane"><div class="lane-label">diff</div>
    <canvas id="wD" height="56"></canvas></div>
  <div id="cursor"></div>
</div>

<div class="controls">
  <button id="playA">play A</button>
  <button id="playB">play B</button>
  <button id="stop">stop</button>
  <span class="hint">click a lane to seek · click a region row to loop it · A/B switches source at same position</span>
</div>

<h2>changed regions</h2>
<table id="tbl">
  <thead><tr><th>#</th><th>start</th><th>end</th><th>length</th>
  <th>severity</th><th></th></tr></thead>
  <tbody></tbody>
</table>

<audio id="audA"></audio>
<audio id="audB"></audio>

<script>
const D = __PAYLOAD__;
const dpr = window.devicePixelRatio || 1;
const dur = D.dur;

document.getElementById('labA').textContent = D.nameA;
document.getElementById('labB').textContent = D.nameB;
document.getElementById('meta').textContent =
  `duration ${fmt(dur)} · ${D.regions.length} changed region` +
  (D.regions.length===1?'':'s') +
  (D.offset ? ` · applied offset ${D.offset}s` : '');

const audA = document.getElementById('audA');
const audB = document.getElementById('audB');
audA.src = `data:${D.mimeA};base64,${D.b64A}`;
audB.src = `data:${D.mimeB};base64,${D.b64B}`;

function fmt(t){const m=Math.floor(t/60),s=(t%60).toFixed(2).padStart(5,'0');
  return `${m}:${s}`}

function drawWave(id, env, color){
  const c=document.getElementById(id), r=c.getBoundingClientRect();
  c.width=r.width*dpr; c.height=c.getAttribute('height')*dpr;
  const ctx=c.getContext('2d'); ctx.scale(dpr,dpr);
  const W=r.width, H=c.getAttribute('height')*1, mid=H/2;
  // changed-region shading behind the wave
  ctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--hot-soft');
  for(const rg of D.regions){
    const x0=rg.start/dur*W, x1=rg.end/dur*W;
    ctx.fillRect(x0,0,x1-x0,H);
  }
  ctx.strokeStyle=color; ctx.lineWidth=1; ctx.beginPath();
  const n=env.length, peak=Math.max(...env, .001);
  for(let i=0;i<n;i++){
    const x=i/(n-1)*W, a=env[i]/peak*(mid-6);
    ctx.moveTo(x,mid-a); ctx.lineTo(x,mid+a+0.5);
  }
  ctx.stroke();
}

function drawDiff(){
  const c=document.getElementById('wD'), r=c.getBoundingClientRect();
  c.width=r.width*dpr; c.height=c.getAttribute('height')*dpr;
  const ctx=c.getContext('2d'); ctx.scale(dpr,dpr);
  const W=r.width, H=c.getAttribute('height')*1;
  const n=D.dist.length;
  for(let i=0;i<n;i++){
    const v=D.dist[i]; // 0..1
    const x=i/n*W, w=W/n+0.5;
    ctx.fillStyle=`rgba(224,96,58,${Math.min(1,v)})`;
    ctx.fillRect(x,H*(1-v),w,H*v);
  }
}

const styleA=getComputedStyle(document.documentElement);
function paint(){
  drawWave('wA',D.envA,styleA.getPropertyValue('--wave-a'));
  drawWave('wB',D.envB,styleA.getPropertyValue('--wave-b'));
  drawDiff();
}
paint(); addEventListener('resize',paint);

// ---- transport
let cur=audA, loop=null;
const cursor=document.getElementById('cursor');
const tl=document.getElementById('timeline');
const bA=document.getElementById('playA'), bB=document.getElementById('playB');

function setSource(a){
  const t=cur.currentTime, playing=!cur.paused;
  cur.pause(); cur=a?audA:audB; cur.currentTime=t;
  bA.classList.toggle('active',a); bB.classList.toggle('active',!a);
  if(playing) cur.play();
}
bA.onclick=()=>{ if(cur===audA&&!cur.paused){return} setSource(true); cur.play(); };
bB.onclick=()=>{ if(cur===audB&&!cur.paused){return} setSource(false); cur.play(); };
document.getElementById('stop').onclick=()=>{cur.pause();loop=null;};

tl.addEventListener('click',e=>{
  const r=tl.getBoundingClientRect();
  const t=(e.clientX-r.left)/r.width*dur;
  loop=null; cur.currentTime=Math.max(0,Math.min(dur,t)); cur.play();
});

setInterval(()=>{
  if(cur.paused){cursor.style.display='none';return}
  cursor.style.display='block';
  const r=tl.getBoundingClientRect();
  cursor.style.left=(cur.currentTime/dur*r.width)+'px';
  if(loop && cur.currentTime>=loop[1]) cur.currentTime=loop[0];
},60);

// ---- region table
const tb=document.querySelector('#tbl tbody');
if(D.regions.length===0){
  tb.innerHTML='<tr><td colspan="6" class="none">no meaningful differences found</td></tr>';
}
const maxPeak=Math.max(...D.regions.map(r=>r.peak),0.001);
D.regions.forEach((rg,i)=>{
  const tr=document.createElement('tr');
  tr.className='region';
  tr.innerHTML=`<td>${i+1}</td><td>${fmt(rg.start)}</td><td>${fmt(rg.end)}</td>
    <td>${(rg.end-rg.start).toFixed(2)}s</td>
    <td><span class="sev" style="width:${Math.round(rg.peak/maxPeak*90)+8}px"></span></td>
    <td style="color:var(--dim)">peak ${rg.peak}</td>`;
  tr.onclick=()=>{loop=[rg.start,rg.end];cur.currentTime=rg.start;cur.play();};
  tb.appendChild(tr);
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="perceptual diff for two audio files")
    ap.add_argument("file_a", help="original / previous version")
    ap.add_argument("file_b", help="new / patched version")
    ap.add_argument("-o", "--output", default="audiodiff.html")
    ap.add_argument("--align", action="store_true",
                    help="auto-correct a global time offset between the files")
    ap.add_argument("--no-embed-compress", action="store_true",
                    help="embed raw WAV instead of ffmpeg-compressed ogg")
    args = ap.parse_args()

    a, sr = load_mono(args.file_a)
    b, _ = load_mono(args.file_b)

    offset_s = 0.0
    if args.align:
        lag = align_offset(a, b, sr)
        offset_s = lag / sr
        if lag > 0:
            b = b[lag:]
        elif lag < 0:
            a = a[-lag:]
        print(f"[align] offset detected: {offset_s:+.3f}s", file=sys.stderr)

    A, tA = log_band_energies(a, sr)
    B, _ = log_band_energies(b, sr)
    dist = spectral_distance(A, B)
    times = tA[: len(dist)]

    regions, thr = find_regions(dist, times)
    dur = min(len(a), len(b)) / sr

    # normalize the distance curve for the diff lane (visual only)
    dmax = max(float(dist.max()), thr * 1.5, 1e-6)
    dist_norm = np.clip(dist / dmax, 0, 1)
    # downsample curve for payload size
    if len(dist_norm) > 2000:
        idx = np.linspace(0, len(dist_norm) - 1, 2000).astype(int)
        dist_norm = dist_norm[idx]

    env_a = peak_envelope(a[: int(dur * sr)])
    env_b = peak_envelope(b[: int(dur * sr)])

    prefer = not args.no_embed_compress
    audio_a = encode_audio(args.file_a, prefer)
    audio_b = encode_audio(args.file_b, prefer)

    html = build_html(
        os.path.basename(args.file_a), os.path.basename(args.file_b),
        dur, env_a, env_b, dist_norm, regions, audio_a, audio_b, offset_s)

    with open(args.output, "w") as fh:
        fh.write(html)

    print(f"{len(regions)} changed region(s), report -> {args.output}")
    for i, r in enumerate(regions, 1):
        print(f"  {i}. {fmt_time(r['start'])} - {fmt_time(r['end'])}"
              f"  (peak {r['peak']})")


if __name__ == "__main__":
    main()
