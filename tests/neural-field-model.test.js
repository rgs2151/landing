import test from 'node:test';
import assert from 'node:assert/strict';
import { AR_COEFFICIENTS, LAG, SIGNAL_WINDOW, createNeuralFieldModel, waveformPath } from '../src/shared/neuralFieldModel.js';

const { signal, points } = createNeuralFieldModel();

test('AR(4) recurrence preserves its two sustained modes throughout the loop', () => {
  for (let n = 4; n < signal.length; n++) {
    const expected = Math.cos(0.095 * n) + 0.6 * Math.cos(0.095 * Math.sqrt(5) * n + 0.7);
    const recurrence = AR_COEFFICIENTS.reduce((sum, coefficient, j) => sum + coefficient * signal[n - j - 1], 0);
    assert.ok(Math.abs(signal[n] - expected) < 1e-7);
    assert.equal(signal[n], recurrence);
  }
});

test('lag-recovered phases reproduce the analytic torus trajectory', () => {
  for (let n = 3 * LAG; n < signal.length; n++) {
    const a = 0.095 * n;
    const b = 0.095 * Math.sqrt(5) * n + 0.7;
    const expected = [(1.4 + 0.5 * Math.cos(b)) * Math.cos(a), 0.5 * Math.sin(b), (1.4 + 0.5 * Math.cos(b)) * Math.sin(a)];
    expected.forEach((value, axis) => assert.ok(Math.abs(points[n * 3 + axis] - value) < 1e-6));
  }
});

test('waveform is finite at playback boundaries and moves with time', () => {
  const start = SIGNAL_WINDOW + 3 * LAG;
  for (const n of [start, start + 1, signal.length - 1]) {
    const path = waveformPath(signal, n);
    assert.equal(path.match(/[ML]/g).length, SIGNAL_WINDOW);
    assert.ok(!/NaN|Infinity/.test(path));
  }
  assert.notEqual(waveformPath(signal, start), waveformPath(signal, start + 1));
});
