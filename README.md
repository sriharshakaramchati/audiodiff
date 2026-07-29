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
- a transform lane: the recovered frequency response of any global effect
  separating the two versions (only shown when one is detected)
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

## Recovered effect curves

Before diffing, audiodiff measures the *global* transform separating the two
versions: a pitch shift (cross-correlating the mean spectra along a log
frequency axis) and a static frequency response (the median per-FFT-bin B/A
magnitude ratio, which a real edit cannot move because it only touches a
minority of frames).

Both used to be a problem to work around. Now they are reported. Anything
global is a *fingerprint* of what was done to the file, not noise to be
suppressed, so the recovered response is printed and drawn:

```
$ python audiodiff.py v1.wav mastered.wav -o report.html
[transform] static effect detected, response from -4.2 dB to +7.8 dB
```

The report grows a `transform` lane under the diff lane, plotting that
response in dB against log frequency against a 0 dB reference. Bounce a track
through a mastering chain and you can read its EQ curve straight off the
report. The header meta line notes the pitch shift and whether a static
effect was found.

The transform is still compensated before the diff runs, so the regions below
it are the residual: what actually changed in the music, with the global
effect accounted for rather than smeared across every region. When nothing
global is detected the signals are left untouched and no lane appears, so
identical files and plain edits diff exactly as before.

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

To see a recovered effect curve, put a global transform on one side:

```
ffmpeg -i examples/v2.wav -af \
  "equalizer=f=100:t=q:w=1:g=8,equalizer=f=3000:t=q:w=1:g=-4" v2_eq.wav
python audiodiff.py examples/v1.wav v2_eq.wav -o report.html
```

The transform lane comes back reading about +8 dB at 100 Hz and -4 dB at
3 kHz, and the edited regions are still found underneath it.

## Ideas / not done yet

- per-stem diffing (run demucs first, diff each stem, one report)
- diff against a specific layer file instead of the full mix
- structural awareness: report changes in bars/beats instead of seconds
- CI mode: exit nonzero if changed regions exceed a threshold, for
  git-hook style checks on music repos
