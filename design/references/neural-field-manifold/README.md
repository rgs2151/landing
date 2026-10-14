# NeuralFieldManifold Animation

`figure-1b.png` is the exact reference supplied by Rudramani Singha. Preserve
its signal / upper hidden-dynamics comparison / lower autoregressive path /
right-hand manifold organization. Omit the panel letter. Dark red: `#8b1e1e`.

The reusable implementation is in `src/shared/NeuralFieldAnimation.jsx`;
the signal model is in `src/shared/neuralFieldModel.js`.

## Demonstration Model

This is an illustrative synthetic signal, not an experimental LFP recording
or a fitted result from the paper. Two incommensurate, sustained oscillatory
modes define a noiseless AR(4) process. Its recurrence is obtained by multiplying
the two second-order oscillator characteristic polynomials. Innovations are
zero in this demonstration; the displayed general equation retains epsilon.

The waveform and moving delay markers use that same recurrence. Four delayed
observations (lag = 8 samples) recover the two modal phases by a linear inverse.
Their product of circles is displayed as a torus with a growing trajectory.
The torus is a phase-coordinate visualization of the delay state, not a claim
that every AR process or its first three raw delay coordinates forms a torus.
The upper hidden-dynamics panel is the static conceptual comparison.

SVG renders labels, equations, arrows, and the waveform. Three.js renders the
manifold; a canvas projection is used if WebGL is unavailable. Animation stops
offscreen, in hidden tabs, and behind an open project. Reduced-motion users see
a static state until they explicitly play it. No recording is downloaded.

## Prototype

Open `/prototype/a` in the Vite development server. Each project opens its
shared title, animation, links, authors, and frozen blurb. Prototype A is still
development-only; the production homepage is unchanged.
