# audiodiff

A perceptual diff for two versions of an audio file. One script, no config.

Built for layer-by-layer AI music workflows: you generated a patch, spliced it
into the song, and want to see and hear exactly what changed against the
previous version. Chunk-level diffs on the binary blobs tell you *that* bytes
changed. This tells you *where in the music* and *how much*.

Try it in the browser (no install, files never leave your machine):
drop two versions of a track at the hosted page, or run the CLI:

```
python audiodiff.py old.wav new.wav -o report.html
```

The browser version lives in `docs/` and is a full client-side port of the
same analysis (verified to produce matching regions on the same inputs).

Output is a single self-contained HTML file (audio embedded, works offline,
shareable over WhatsApp) with:

- both versions rendered as DAW-style waveform lanes
- a diff lane: per-window spectral distance between the two files
- changed regions detected automatically, listed with timestamps and severity
- click a region row to loop it, use play A / play B to switch source at the
  same playhead position for instant A/B comparison

## How it works

Both files are downmixed to mono at 22.05 kHz, converted to log-spaced band
energies per STFT frame (48 bands, 40 Hz to Nyquist), and compared frame by
frame with a blend of normalized L2 (level changes) and cosine distance
(timbre changes). The curve is smoothed over ~250 ms and thresholded with a
median + MAD rule, so the sensitivity adapts to each pair of files. Identical
files produce zero regions.

## Flags

```
--align                auto-correct a global time offset between the files
                       (onset-envelope cross-correlation, max 5s)
--no-embed-compress    embed raw WAV instead of ffmpeg-compressed ogg
-o, --output           output path (default audiodiff.html)
```

## Install

```
pip install numpy scipy soundfile
```

ffmpeg is optional but recommended, it keeps report size small by embedding
ogg instead of wav.

## Demo

```
cd examples
python make_demo.py          # synthesizes v1.wav and v2.wav
cd ..
python audiodiff.py examples/v1.wav examples/v2.wav -o report.html
```

The demo track pair mimics a patch workflow: v2 adds an arp layer at 8-14s
and regenerates the lead at 20-25s. The diff finds exactly those two regions.

## Ideas / not done yet

- per-stem diffing (run demucs first, diff each stem, one report)
- diff against a specific layer file instead of the full mix
- structural awareness: report changes in bars/beats instead of seconds
- CI mode: exit nonzero if changed regions exceed a threshold, for
  git-hook style checks on music repos
