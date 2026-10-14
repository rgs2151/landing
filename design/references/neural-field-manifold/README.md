# NeuralFieldManifold Animation

`figure-1b.png` is the exact reference supplied by Rudramani Singha. Preserve
its signal / upper hidden-dynamics comparison / lower autoregressive path /
right-hand manifold organization. Omit the panel letter. Dark red: `#8b1e1e`.

The reusable implementation is in `src/shared/NeuralFieldAnimation.jsx`;
the signal model is in `src/shared/neuralFieldModel.js`.

## Recorded Mouse LFP

The animation uses the real processed mouse hippocampal LFP from the
[mouse LFP tutorial](https://github.com/erfanzabeh/NeuralFieldManifold/blob/9eafbedcfbddb25f0d6102db6e75ddfa6eb31c79/docs/tutorials/mouse_lfp.ipynb),
not a simulation or AR prediction. The existing local notebook cache is
`/home/dev/NeuralFieldManifold/notebooks/rudra_novak/data/mice_lfp.npz`.

- Source recording: `eke003_spont_250818_175844.npy`, channel index 1.
- First five seconds: 2,500 processed samples at 500 Hz.
- The cache's own lag: 10 samples (20 ms), embedding dimension 3.
- Tutorial processing: 50 Hz notch (Q=30), 1-20 Hz bandpass,
  downsampling from 20,000 to 500 Hz, then the tutorial's 1-80 Hz
  envelope normalization with a 3 Hz envelope lowpass.
- Cached `xs` values are exported unchanged to
  `public/assets/data/mouse-lfp.json` (47,791 bytes).

The JSON preserves every original float64 value. Its metadata records the
source, parameters and SHA-256 of the little-endian float64 sample bytes.
The browser fetches it once, shares it across views, and replays at 500 samples
per second in recorded time, displayed at half speed (250 samples per wall-clock
second). Playback and replay start midway at sample 1,250 with history already
populated. The waveform shows one recorded second and the trailing embedding
shows 1,000 valid delay states. History wraps without clearing; separate line
segments avoid inventing a connection between the recording's endpoints.
The cursor cycles through valid delay states 20-2,499; the first 20 samples
remain the lag context. No smoothing, crossfade or synthesized samples are used.

Regenerate only from the original cache:

```sh
python scripts/cache-mouse-lfp.py /path/to/mice_lfp.npz
```

This exporter needs NumPy. Only the small processed excerpt is included in the
website; the raw recording, other repository and notebooks remain outside it.

The displayed coordinates are x(t), x(t-tau), x(t-2tau), uniformly scaled for
viewing by a single constant gain. No modal-phase recovery, imposed torus surface, or radial projection is
used. The waveform, lag markers and moving 3D point share the same sample index.
The upper hidden-dynamics panel remains the static conceptual comparison.

SVG renders labels, equations, arrows, and the waveform. Three.js renders the
manifold; a canvas projection is used if WebGL is unavailable. Animation stops
offscreen, in hidden tabs, and behind an open project. Reduced-motion users see
a static state until they explicitly play it. The source raw recording is never
downloaded by the browser.

## Prototype

Open `/prototype/a` in the Vite development server. Each project opens its
shared title, animation, links, authors, and frozen blurb. Prototype A is still
development-only; the production homepage is unchanged.
