export const RECORDING_URL = '/assets/data/mouse-lfp.json';
export const SIGNAL_WINDOW = 500;
export const TRACE_LENGTH = 1000;
export const SIGNAL_HEIGHT = 50;
export const POINT_SCALE = 1.8;

export function createNeuralFieldModel(recording) {
  const { samples, sampleRate, lagSamples: lag } = recording;
  if (recording.kind !== 'recorded-mouse-lfp' || !Array.isArray(samples)
    || samples.length !== 2500 || !samples.every(Number.isFinite)
    || sampleRate !== 500 || lag !== 10) {
    throw new Error('Invalid mouse LFP recording');
  }
  const peak = samples.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0);
  if (!peak) throw new Error('Empty mouse LFP signal');
  // Only a constant display gain; the cached processed samples remain untouched.
  const signal = Float64Array.from(samples, value => value / peak);
  const points = new Float32Array(signal.length * 3);
  for (let n = 2 * lag; n < signal.length; n++) {
    points.set([signal[n], signal[n - lag], signal[n - 2 * lag]].map(x => x * POINT_SCALE), n * 3);
  }
  return { signal, points, samples, sampleRate, lag };
}

let cachedModel;
export function loadNeuralFieldModel() {
  cachedModel ??= fetch(RECORDING_URL)
    .then(response => {
      if (!response.ok) throw new Error('Could not load mouse LFP recording');
      return response.json();
    })
    .then(createNeuralFieldModel)
    .catch(error => { cachedModel = undefined; throw error; });
  return cachedModel;
}

export function advanceFrame(frame, steps, length) {
  return (frame + steps) % length;
}

export function waveformPath(signal, frame) {
  const count = Math.min(SIGNAL_WINDOW, frame + 1);
  return Array.from({ length: count }, (_, i) => {
    const x = 18 + (SIGNAL_WINDOW - count + i) * 224 / (SIGNAL_WINDOW - 1);
    const y = 286 - signal[frame - count + 1 + i] * SIGNAL_HEIGHT;
    return `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
}
