// Port of notebooks/rudra_novak/donuts_retry_improve.ipynb, cells 1, 2, 8.
export const SIMULATION = Object.freeze({
  samples: 5000, burnin: 5000, seed: 42, noise: 0.05, sampleRate: 100,
  rho1: 0.999, frequency1: 0.025, rho2: 0.993, frequency2: 0.10,
});
export const LAG = 10;
export const SIGNAL_WINDOW = 500;
export const TRACE_LENGTH = 1500 - 2 * LAG;
export const SIGNAL_HEIGHT = 50;
export const POINT_SCALE = 1.8;
const b1 = 2 * SIMULATION.rho1 * Math.cos(2 * Math.PI * SIMULATION.frequency1);
const b2 = 2 * SIMULATION.rho2 * Math.cos(2 * Math.PI * SIMULATION.frequency2);
const c1 = SIMULATION.rho1 ** 2;
const c2 = SIMULATION.rho2 ** 2;
export const AR_COEFFICIENTS = [b1 + b2, -(c1 + c2 + b1 * b2), b1 * c2 + b2 * c1, -c1 * c2];

// Same Gaussian distribution as the notebook, not NumPy's PCG64 bitstream.
function gaussianRandom(seed) {
  const uniform = () => {
    seed |= 0;
    seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(Math.max(uniform(), 1e-10))) * Math.cos(2 * Math.PI * uniform());
}

export function createNeuralFieldModel(length = SIMULATION.samples, seed = SIMULATION.seed) {
  const random = gaussianRandom(seed);
  const raw = new Float64Array(length + SIMULATION.burnin);
  for (let n = 0; n < 4; n++) raw[n] = random();
  for (let n = 4; n < raw.length; n++) {
    raw[n] = AR_COEFFICIENTS.reduce((sum, coefficient, j) => sum + coefficient * raw[n - j - 1], 0)
      + SIMULATION.noise * random();
  }
  const samples = raw.slice(SIMULATION.burnin);
  const minimum = samples.reduce((a, b) => Math.min(a, b), Infinity);
  const maximum = samples.reduce((a, b) => Math.max(a, b), -Infinity);
  const signal = samples.map(value => (value - minimum) / (maximum - minimum) * 2 - 1);
  const points = new Float32Array(length * 3);
  for (let n = 2 * LAG; n < length; n++) {
    // Actual delay coordinates, not a prescribed torus or recovered phases.
    points.set([signal[n], signal[n - LAG], signal[n - 2 * LAG]].map(x => x * POINT_SCALE), n * 3);
  }
  return { signal, points, samples };
}

export function waveformPath(signal, frame) {
  return Array.from({ length: SIGNAL_WINDOW }, (_, i) => {
    const x = 18 + i * 224 / (SIGNAL_WINDOW - 1);
    const y = 286 - signal[frame - SIGNAL_WINDOW + 1 + i] * SIGNAL_HEIGHT;
    return `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
}
