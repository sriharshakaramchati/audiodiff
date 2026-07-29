"""Synthesizes v1.wav and v2.wav of a fake 30s track.

v2 differs from v1 in two places, mimicking a layer-patch workflow:
  - 8s-14s:  a new arp layer was added
  - 20s-25s: the lead melody was regenerated (different notes + timbre)
Everything else is identical, so a good diff should flag exactly
those two regions and nothing else.
"""
import numpy as np
import soundfile as sf

SR = 44100
DUR = 30.0
t = np.arange(int(SR * DUR)) / SR


def env_ad(sig, seg_len, a=0.01, d=0.3):
    n = len(sig)
    e = np.ones(n)
    an, dn = int(a * SR), int(d * SR)
    per = int(seg_len * SR)
    for s in range(0, n, per):
        e[s:s + an] = np.linspace(0, 1, min(an, n - s))
        tail = slice(s + an, min(s + per, n))
        L = tail.stop - tail.start
        if L > 0:
            e[tail] = np.exp(-np.linspace(0, 5, L) / (d * 5))
    return sig * e


def bass(t):
    notes = [55, 55, 65.4, 49]  # A1 A1 C2 G1
    f = np.zeros_like(t)
    bar = 2.0
    for i in range(len(t)):
        f[i] = notes[int(t[i] // bar) % 4]
    sig = 0.5 * np.sign(np.sin(2 * np.pi * f * t)) * 0.3 + \
        0.7 * np.sin(2 * np.pi * f * t)
    return env_ad(sig, 0.5) * 0.5


def drums(t):
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(t))
    sig = np.zeros_like(t)
    step = int(0.5 * SR)  # 120bpm eighths
    for s in range(0, len(t), step):
        L = min(int(0.08 * SR), len(t) - s)
        sig[s:s + L] += noise[s:s + L] * np.exp(-np.linspace(0, 8, L)) * 0.4
    kick_step = int(1.0 * SR)
    for s in range(0, len(t), kick_step):
        L = min(int(0.15 * SR), len(t) - s)
        ph = np.cumsum(np.linspace(120, 50, L)) / SR
        sig[s:s + L] += np.sin(2 * np.pi * ph) * np.exp(-np.linspace(0, 6, L)) * 0.9
    return sig * 0.6


def lead(t, seed=1, base=440.0):
    rng = np.random.default_rng(seed)
    scale = base * 2 ** (np.array([0, 2, 3, 5, 7, 10]) / 12)
    f = np.zeros_like(t)
    per = int(0.5 * SR)
    seq = rng.choice(scale, size=len(t) // per + 1)
    for i, fr in enumerate(seq):
        f[i * per:(i + 1) * per] = fr
    vib = 1 + 0.005 * np.sin(2 * np.pi * 5 * t)
    sig = np.sin(2 * np.pi * np.cumsum(f * vib) / SR)
    sig += 0.3 * np.sin(2 * np.pi * 2 * np.cumsum(f * vib) / SR)
    return env_ad(sig, 0.5) * 0.25


def arp(t, base=880.0):
    scale = base * 2 ** (np.array([0, 3, 7, 12]) / 12)
    f = np.zeros_like(t)
    per = int(0.125 * SR)
    for i in range(len(t) // per + 1):
        f[i * per:(i + 1) * per] = scale[i % 4]
    sig = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))
    return env_ad(sig, 0.125) * 0.12


mix1 = bass(t) + drums(t) + lead(t, seed=1)

# v2: same core, add arp 8-14s, regenerate lead 20-25s
mix2 = bass(t) + drums(t)
lead1 = lead(t, seed=1)
lead2 = lead(t, seed=42, base=523.25)  # different notes + register
m_keep = np.ones_like(t)
m_keep[(t >= 20) & (t < 25)] = 0
xf = int(0.05 * SR)
m_keep = np.convolve(m_keep, np.ones(xf) / xf, mode="same")  # soft crossfade
mix2 += lead1 * m_keep + lead2 * (1 - m_keep)
a = arp(t)
m_arp = np.zeros_like(t)
m_arp[(t >= 8) & (t < 14)] = 1
m_arp = np.convolve(m_arp, np.ones(xf) / xf, mode="same")
mix2 += a * m_arp

for name, mix in [("v1.wav", mix1), ("v2.wav", mix2)]:
    mix = mix / np.max(np.abs(mix)) * 0.9
    sf.write(name, mix.astype(np.float32), SR)
    print("wrote", name)
