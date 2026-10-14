# NeuralFieldManifold Animation

`figure-1b.png` is the exact reference supplied by Rudramani Singha. Preserve
its signal / upper hidden-dynamics comparison / lower autoregressive path /
right-hand manifold organization. Omit the panel letter. Dark red: `#8b1e1e`.

The reusable implementation is in `src/shared/NeuralFieldAnimation.jsx`;
the signal model is in `src/shared/neuralFieldModel.js`.

## Demonstration Model

The simulation now follows `notebooks/rudra_novak/donuts_retry_improve.ipynb`,
cells 1, 2 and 8, in [NeuralFieldManifold](https://github.com/erfanzabeh/NeuralFieldManifold/blob/9eafbedcfbddb25f0d6102db6e75ddfa6eb31c79/notebooks/rudra_novak/donuts_retry_improve.ipynb).
The reference clone stays outside this repository, at
`/tmp/landing-NeuralFieldManifold-reference`; no notebook or dataset is vendored.

- Noise-driven AR(4), with pole radii 0.999 and 0.993.
- Frequencies 0.025 and 0.10 cycles/sample; sampling rate 100 Hz.
- Gaussian innovation standard deviation 0.05, 5,000-sample burn-in,
  5,000 retained samples, seed 42.
- The notebook's min/max normalization to [-1, 1].
- Direct three-coordinate delay embedding with lag 10; the initial trajectory
  uses the first 1,500 samples, as in the notebook's low-noise AR(4) figure.

JavaScript uses a seeded Mulberry32/Box-Muller Gaussian stream, not NumPy PCG64:
the model and parameters match, but individual random samples are not bitwise
identical. This remains synthetic data, not an experimental recording.

The displayed coordinates are x(t), x(t-tau), x(t-2tau), uniformly scaled for
viewing. No modal-phase recovery, imposed torus surface, or radial projection is
used. The waveform, lag markers and moving 3D point share the same sample index.
The upper hidden-dynamics panel remains the static conceptual comparison.

SVG renders labels, equations, arrows, and the waveform. Three.js renders the
manifold; a canvas projection is used if WebGL is unavailable. Animation stops
offscreen, in hidden tabs, and behind an open project. Reduced-motion users see
a static state until they explicitly play it. No recording is downloaded.

## Prototype

Open `/prototype/a` in the Vite development server. Each project opens its
shared title, animation, links, authors, and frozen blurb. Prototype A is still
development-only; the production homepage is unchanged.
