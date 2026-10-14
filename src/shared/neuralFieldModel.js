export const RECORDING_URL = '/assets/data/mouse-lfp.json';
export const SIGNAL_WINDOW = 500;
export const TRACE_LENGTH = 1000;
export const SIGNAL_HEIGHT = 50;
export const POINT_SCALE = 1.8;
export const PLAYBACK_SPEED = 0.125;

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

export function advanceFrame(frame, steps, length, first = 0) {
  return first + (frame - first + steps) % (length - first);
}

export function traceRanges(frame, length, lag) {
  const first = 2 * lag;
  const start = frame - TRACE_LENGTH + 1;
  if (start >= first) return [{ start, count: TRACE_LENGTH }];
  // Keep the previous pass visible without drawing across the recording cut.
  return [
    { start: length - (first - start), count: first - start },
    { start: first, count: frame - first + 1 },
  ];
}

export function waveformPath(signal, frame) {
  return Array.from({ length: SIGNAL_WINDOW }, (_, i) => {
    const index = (frame - SIGNAL_WINDOW + 1 + i + signal.length) % signal.length;
    const x = 18 + i * 224 / (SIGNAL_WINDOW - 1);
    const y = 286 - signal[index] * SIGNAL_HEIGHT;
    return `${i && index !== 0 ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(' ');
}
