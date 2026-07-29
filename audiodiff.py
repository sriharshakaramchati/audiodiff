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
  - a transform lane showing the recovered response of any global
    effect that separates the two versions
  - a list of changed regions with timestamps and severity
  - embedded audio players; click any region to A/B it

Global transforms (a pitch shift, a static EQ or level change) are
detected, measured and reported rather than silently cancelled: the
recovered frequency response is drawn in the report, and the diff below
it is the residual once that transform is accounted for.

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
import warnings

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

# global-transform detection
PITCH_STEP = 0.05        # semitones per log-frequency grid step
PITCH_MAX = 12.0         # widest shift we look for, semitones
PITCH_MIN_REPORT = 0.25  # below this, call it zero
PITCH_MIN_MARGIN = 1.02  # shifted fit must beat the no-shift fit by this much
PITCH_MIN_FIT = 0.30     # and must be at least this good in absolute terms
FINE_PER_OCTAVE = 96     # fine log-frequency grid used to undo a pitch shift
GAIN_FLOOR_FRAC = 0.20   # a bin only votes on gain when A is this loud
GAIN_MIN_VOTES = 8       # bins with fewer voting frames stay at unity
EFFECT_POINTS = 200      # points on the reported response curve
EFFECT_DB_CLIP = 24.0    # response is clipped to +/- this many dB
EFFECT_DB_TOL = 1.5      # a point counts as coloured past this deviation
EFFECT_MIN_FRAC = 0.10   # fraction of coloured points that trips detection


def band_edges(sr):
    return np.geomspace(40.0, sr / 2 * 0.95, N_BANDS + 1)


def stft_mag(x, sr):
    """Magnitude spectrogram -> (freqs, times, mag) with mag (freqs, frames)."""
    f, t, Z = sps.stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                       padded=False, boundary=None)
    return f, t, np.abs(Z)


def bands_from_mag(freqs, mag, sr):
    """Average a magnitude spectrogram into log-spaced bands.
    Works for any ascending frequency axis, linear bins or log grid."""
    edges = band_edges(sr)
    bands = np.zeros((mag.shape[1], N_BANDS), dtype=np.float32)
    for i in range(N_BANDS):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if sel.any():
            bands[:, i] = mag[sel].mean(axis=0)
    return np.log1p(bands * 100.0)


def log_band_energies(x, sr):
    """Log-spaced band energies per STFT frame -> (frames, N_BANDS)."""
    f, t, mag = stft_mag(x, sr)
    return bands_from_mag(f, mag, sr), t


def interp_weights(src_f, dst_f):
    """Index/weight pair for linearly resampling a spectrum from the
    ascending axis src_f onto dst_f (clamped at both ends)."""
    dst = np.clip(dst_f, src_f[0], src_f[-1])
    idx = np.clip(np.searchsorted(src_f, dst, side="left"), 1, len(src_f) - 1)
    f0, f1 = src_f[idx - 1], src_f[idx]
    w = np.clip((dst - f0) / np.maximum(f1 - f0, 1e-12), 0.0, 1.0)
    return idx, w


def interp_spectra(mag, idx, w):
    """Apply interp_weights to every frame of a (freqs, frames) matrix."""
    return mag[idx - 1] * (1.0 - w)[:, None] + mag[idx] * w[:, None]


def fine_axis(sr):
    """Fine log-frequency grid, where a pitch shift is a pure translation."""
    lo, hi = 40.0, sr / 2 * 0.95
    n = int(round(np.log2(hi / lo) * FINE_PER_OCTAVE)) + 1
    return lo * 2.0 ** (np.arange(n) / FINE_PER_OCTAVE)


def estimate_pitch_shift(freqs, mag_a, mag_b):
    """Global pitch shift of B relative to A, in semitones (positive: B is
    higher). A pitch shift translates the spectrum along log frequency, so
    the mean spectra are cross-correlated on a log axis. Returns 0.0 when
    there is no shift worth reporting."""
    lo, hi = 60.0, freqs[-1] * 0.9
    if hi <= lo * 2:
        return 0.0
    n = int(np.log2(hi / lo) * 12.0 / PITCH_STEP) + 1
    grid = lo * 2.0 ** (np.arange(n) * PITCH_STEP / 12.0)
    idx, w = interp_weights(freqs, grid)
    sa = np.log1p(interp_spectra(mag_a.mean(axis=1)[:, None], idx, w)[:, 0] * 100.0)
    sb = np.log1p(interp_spectra(mag_b.mean(axis=1)[:, None], idx, w)[:, 0] * 100.0)
    # score every candidate shift by the correlation over its overlap only.
    # a plain dot product would grow with overlap length and so always favour
    # zero lag; normalizing per lag removes that bias.
    best_lag, best_fit, flat_fit = 0, -1.0, -1.0
    for lag in range(-int(PITCH_MAX / PITCH_STEP), int(PITCH_MAX / PITCH_STEP) + 1):
        u = sa[: n - lag] if lag >= 0 else sa[-lag:]
        v = sb[lag:] if lag >= 0 else sb[: n + lag]
        if len(u) < n // 2:
            continue
        u, v = u - u.mean(), v - v.mean()
        den = np.linalg.norm(u) * np.linalg.norm(v)
        fit = float(np.dot(u, v) / den) if den > 1e-12 else 0.0
        if lag == 0:
            flat_fit = fit
        if fit > best_fit:
            best_lag, best_fit = lag, fit
    semitones = float(best_lag * PITCH_STEP)
    if abs(semitones) < PITCH_MIN_REPORT or best_fit < PITCH_MIN_FIT:
        return 0.0
    if best_fit <= flat_fit * PITCH_MIN_MARGIN:
        return 0.0
    return semitones


def bin_gains(mag_a, mag_b):
    """Median per-bin B/A magnitude ratio. This is the frequency response
    of whatever static effect separates the two versions: a real edit moves
    a minority of frames, so the median across the file sees past it."""
    ref = mag_a.max(axis=1, keepdims=True)
    votes = mag_a > np.maximum(GAIN_FLOOR_FRAC * ref, 1e-6)
    ratio = np.where(votes, mag_b / np.maximum(mag_a, 1e-12), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g = np.nanmedian(ratio, axis=1)
    g[(votes.sum(axis=1) < GAIN_MIN_VOTES) | ~np.isfinite(g)] = 1.0
    return g


def normalize_gain_bins(freqs, mag_a, mag_b):
    """Recover the static response per FFT bin and line A up with B.
    Returns (mag_a_matched, mag_b, gain, gain_freqs)."""
    g = bin_gains(mag_a, mag_b)
    return mag_a * g[:, None], mag_b, g, freqs


def normalize_gain_fine(freqs, mag_a, mag_b, semitones, sr):
    """Same, on the fine log-frequency grid and after undoing a pitch shift:
    B is read at freq * 2**(semitones/12) so its partials land on A's."""
    axis = fine_axis(sr)
    warped = axis * 2.0 ** (semitones / 12.0)
    fine_a = interp_spectra(mag_a, *interp_weights(freqs, axis))
    fine_b = interp_spectra(mag_b, *interp_weights(freqs, warped))
    g = bin_gains(fine_a, fine_b)
    # a shift up walks the top of the axis past B's Nyquist, where there is
    # no evidence either way: leave those points flat rather than inventing a
    # response out of clamped edge bins
    g[(warped > freqs[-1]) | (warped < freqs[0])] = 1.0
    return fine_a * g[:, None], fine_b, g, axis


def effect_response(gain, gain_freqs, sr):
    """Recovered gain curve as dB on a ~200 point log-frequency axis, plus
    whether it is coloured enough to be worth reporting."""
    db = np.clip(20.0 * np.log10(np.maximum(gain, 1e-6)),
                 -EFFECT_DB_CLIP, EFFECT_DB_CLIP)
    axis = np.geomspace(40.0, sr / 2 * 0.95, EFFECT_POINTS)
    half = 0.5 * np.log(axis[-1] / axis[0]) / (EFFECT_POINTS - 1)
    idx, w = interp_weights(gain_freqs, axis)
    curve = np.empty(EFFECT_POINTS)
    for i, fc in enumerate(axis):
        # average the native-resolution response over each log cell; where a
        # cell falls between source bins, interpolate instead
        sel = (gain_freqs >= fc * np.exp(-half)) & (gain_freqs < fc * np.exp(half))
        if sel.any():
            curve[i] = db[sel].mean()
        else:
            curve[i] = db[idx[i] - 1] * (1.0 - w[i]) + db[idx[i]] * w[i]
    detected = bool(np.mean(np.abs(curve) > EFFECT_DB_TOL) > EFFECT_MIN_FRAC)
    return curve, axis, detected


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
               audio_a, audio_b, offset_s, semitones, effect_db, effect_hz,
               effect):
    payload = {
        "nameA": name_a, "nameB": name_b, "dur": round(dur, 3),
        "envA": [round(float(v), 4) for v in env_a],
        "envB": [round(float(v), 4) for v in env_b],
        "dist": [round(float(v), 4) for v in dist_norm],
        "regions": regions,
        "mimeA": audio_a[0], "b64A": audio_a[1],
        "mimeB": audio_b[0], "b64B": audio_b[1],
        "offset": round(offset_s, 3),
        "pitch": round(float(semitones), 2),
        "effect": bool(effect),
        "effectDb": [round(float(v), 2) for v in effect_db] if effect else [],
        "effectHz": [round(float(v), 1) for v in effect_hz] if effect else [],
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

<div class="lane" id="laneT"><div class="lane-label">transform</div>
  <canvas id="wT" height="72"></canvas></div>

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
  (D.offset ? ` · applied offset ${D.offset}s` : '') +
  (D.pitch ? ` · pitch shift ${D.pitch>0?'+':''}${D.pitch} st` : '') +
  (D.effect ? ' · static effect detected' : '');

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

// recovered response of the detected global effect, dB against log frequency
function drawTransform(){
  const c=document.getElementById('wT'), r=c.getBoundingClientRect();
  c.width=r.width*dpr; c.height=c.getAttribute('height')*dpr;
  const ctx=c.getContext('2d'); ctx.scale(dpr,dpr);
  const W=r.width, H=c.getAttribute('height')*1;
  const db=D.effectDb, hz=D.effectHz, n=db.length;
  const cs=getComputedStyle(document.documentElement);
  const line=cs.getPropertyValue('--line').trim();
  const dim=cs.getPropertyValue('--dim').trim();
  const hot=cs.getPropertyValue('--hot').trim();
  const soft=cs.getPropertyValue('--hot-soft').trim();
  const mono=cs.getPropertyValue('--mono').trim();
  const pad=14;
  let span=6;
  for(const v of db) span=Math.max(span,Math.ceil(Math.abs(v)));
  const X=i=>i/(n-1)*W, Y=v=>pad+(0.5-v/(2*span))*(H-2*pad);
  // log-frequency gridlines
  ctx.font='9px '+mono; ctx.textBaseline='alphabetic';
  const lr=Math.log(hz[n-1]/hz[0]);
  for(const f of [100,1000,10000]){
    if(f<hz[0]||f>hz[n-1]) continue;
    const x=Math.log(f/hz[0])/lr*W;
    ctx.strokeStyle=line; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
    ctx.fillStyle=dim;
    ctx.fillText(f>=1000?(f/1000)+'k':''+f, x+3, H-5);
  }
  // 0 dB reference and scale
  ctx.strokeStyle=line; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(0,Y(0)); ctx.lineTo(W,Y(0)); ctx.stroke();
  ctx.fillStyle=dim; ctx.textAlign='right';
  ctx.fillText('+'+span+' dB', W-6, Y(span)+8);
  ctx.fillText('-'+span+' dB', W-6, Y(-span)-3);
  ctx.textAlign='left';
  // curve, with the deviation from flat shaded in
  ctx.beginPath(); ctx.moveTo(X(0),Y(0));
  for(let i=0;i<n;i++) ctx.lineTo(X(i),Y(db[i]));
  ctx.lineTo(X(n-1),Y(0)); ctx.closePath();
  ctx.fillStyle=soft; ctx.fill();
  ctx.beginPath();
  for(let i=0;i<n;i++){const x=X(i),y=Y(db[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
  ctx.strokeStyle=hot; ctx.lineWidth=1.5; ctx.stroke();
}

const styleA=getComputedStyle(document.documentElement);
const laneT=document.getElementById('laneT');
if(!D.effect) laneT.remove();
function paint(){
  drawWave('wA',D.envA,styleA.getPropertyValue('--wave-a'));
  drawWave('wB',D.envB,styleA.getPropertyValue('--wave-b'));
  drawDiff();
  if(D.effect) drawTransform();
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

    fA, tA, mag_a = stft_mag(a, sr)
    _, _, mag_b = stft_mag(b, sr)
    nf = min(mag_a.shape[1], mag_b.shape[1])
    mag_a, mag_b = mag_a[:, :nf], mag_b[:, :nf]

    # measure the global transform separating the two versions, then report it
    semitones = estimate_pitch_shift(fA, mag_a, mag_b)
    if semitones:
        cor_a, cor_b, gain, gain_hz = normalize_gain_fine(
            fA, mag_a, mag_b, semitones, sr)
    else:
        cor_a, cor_b, gain, gain_hz = normalize_gain_bins(fA, mag_a, mag_b)
    effect_db, effect_hz, effect = effect_response(gain, gain_hz, sr)

    if semitones:
        print(f"[transform] pitch shift detected: {semitones:+.2f} semitones",
              file=sys.stderr)
    if effect:
        print(f"[transform] static effect detected, response from "
              f"{effect_db.min():+.1f} dB to {effect_db.max():+.1f} dB",
              file=sys.stderr)

    # only compensate when there is something to compensate, so an untouched
    # pair diffs exactly as it would have without this stage
    if semitones or effect:
        A = bands_from_mag(gain_hz, cor_a, sr)
        B = bands_from_mag(gain_hz, cor_b, sr)
    else:
        A = bands_from_mag(fA, mag_a, sr)
        B = bands_from_mag(fA, mag_b, sr)
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
        dur, env_a, env_b, dist_norm, regions, audio_a, audio_b, offset_s,
        semitones, effect_db, effect_hz, effect)

    with open(args.output, "w") as fh:
        fh.write(html)

    print(f"{len(regions)} changed region(s), report -> {args.output}")
    for i, r in enumerate(regions, 1):
        print(f"  {i}. {fmt_time(r['start'])} - {fmt_time(r['end'])}"
              f"  (peak {r['peak']})")


if __name__ == "__main__":
    main()
