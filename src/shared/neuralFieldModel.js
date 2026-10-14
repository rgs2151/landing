import { Matrix4, Vector4 } from 'three';

export const LAG = 8;
export const SIGNAL_WINDOW = 190;
export const TRACE_LENGTH = 720;
const frequencies = [0.095, 0.095 * Math.sqrt(5)];
const [c1, c2] = frequencies.map(Math.cos);
export const AR_COEFFICIENTS = [2 * (c1 + c2), -(2 + 4 * c1 * c2), 2 * (c1 + c2), -1];

// Two persistent oscillatory modes are an exact, noiseless AR(4) process.
export function createNeuralFieldModel(length = 12000) {
  const signal = new Float64Array(length);
  for (let n = 0; n < 4; n++) {
    signal[n] = Math.cos(frequencies[0] * n) + 0.6 * Math.cos(frequencies[1] * n + 0.7);
  }
  for (let n = 4; n < length; n++) {
    signal[n] = AR_COEFFICIENTS.reduce((sum, coefficient, j) => sum + coefficient * signal[n - j - 1], 0);
  }

  // Recover the two modal phases from four lagged observations, then display
  // their product of circles as a torus. This is not a raw 3-delay projection.
  const rows = Array.from({ length: 4 }, (_, j) => frequencies.flatMap(w => [Math.cos(w * j * LAG), Math.sin(w * j * LAG)]));
  const inverse = new Matrix4().set(...rows.flat()).invert();
  const modes = new Vector4();
  const points = new Float32Array(length * 3);
  for (let n = 3 * LAG; n < length; n++) {
    modes.set(signal[n], signal[n - LAG], signal[n - 2 * LAG], signal[n - 3 * LAG]).applyMatrix4(inverse);
    const a = Math.atan2(modes.y, modes.x);
    const b = Math.atan2(modes.w, modes.z);
    const radius = 1.4 + 0.5 * Math.cos(b);
    points.set([radius * Math.cos(a), 0.5 * Math.sin(b), radius * Math.sin(a)], n * 3);
  }
  return { signal, points };
}

export function waveformPath(signal, frame) {
  return Array.from({ length: SIGNAL_WINDOW }, (_, i) => {
    const x = 18 + i * 224 / (SIGNAL_WINDOW - 1);
    const y = 286 - signal[frame - SIGNAL_WINDOW + 1 + i] * 24;
    return `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
}
