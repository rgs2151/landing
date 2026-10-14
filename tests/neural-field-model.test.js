import test from 'node:test';
import assert from 'node:assert/strict';
import { AR_COEFFICIENTS, LAG, SIGNAL_WINDOW, SIMULATION, POINT_SCALE, createNeuralFieldModel, waveformPath } from '../src/shared/neuralFieldModel.js';

const { signal, points, samples } = createNeuralFieldModel();

test('uses the notebook AR(4) poles, burn-in, noise and lag parameters', () => {
  assert.deepEqual(SIMULATION, { samples: 5000, burnin: 5000, seed: 42, noise: 0.05, sampleRate: 100, rho1: 0.999, frequency1: 0.025, rho2: 0.993, frequency2: 0.10 });
  assert.equal(LAG, 10);
  const [a, b] = [2 * .999 * Math.cos(2 * Math.PI * .025), 2 * .993 * Math.cos(2 * Math.PI * .1)];
  assert.deepEqual(AR_COEFFICIENTS, [a + b, -(.999 ** 2 + .993 ** 2 + a * b), a * .993 ** 2 + b * .999 ** 2, -(.999 ** 2 * .993 ** 2)]);
});

test('recurrence is driven by zero-mean Gaussian noise with the notebook variance', () => {
  const innovations = samples.slice(4).map((value, index) => value - AR_COEFFICIENTS.reduce((sum, c, j) => sum + c * samples[index + 3 - j], 0));
  const mean = innovations.reduce((a, b) => a + b) / innovations.length;
  const sd = Math.sqrt(innovations.reduce((sum, x) => sum + (x - mean) ** 2, 0) / innovations.length);
  assert.ok(Math.abs(mean) < .003);
  assert.ok(Math.abs(sd - SIMULATION.noise) < .003);
  assert.equal(Math.min(...signal), -1);
  assert.equal(Math.max(...signal), 1);
  assert.deepEqual(createNeuralFieldModel().signal, signal);
  assert.notDeepEqual(createNeuralFieldModel(5000, 43).signal, signal);
});

test('3D geometry is the direct lag embedding of the displayed signal', () => {
  for (let n = 2 * LAG; n < signal.length; n++) {
    for (let axis = 0; axis < 3; axis++) {
      assert.equal(points[n * 3 + axis], Math.fround(signal[n - axis * LAG] * POINT_SCALE));
    }
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
