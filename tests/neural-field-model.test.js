import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { SIGNAL_WINDOW, POINT_SCALE, createNeuralFieldModel, advanceFrame, waveformPath } from '../src/shared/neuralFieldModel.js';

const recording = JSON.parse(readFileSync(new URL('../public/assets/data/mouse-lfp.json', import.meta.url)));
const { signal, points, samples, lag, sampleRate } = createNeuralFieldModel(recording);

test('cache preserves all 2,500 original processed mouse LFP samples exactly', () => {
  assert.equal(sampleRate, 500);
  assert.equal(lag, 10);
  assert.equal(recording.durationSeconds, 5);
  assert.equal(samples.length, 2500);
  const bytes = Buffer.alloc(samples.length * 8);
  samples.forEach((value, i) => bytes.writeDoubleLE(value, i * 8));
  const checksum = createHash('sha256').update(bytes).digest('hex');
  assert.equal(checksum, 'beacf0a4bac1624a500f43aec43869ef7b6bfceab48bae801eb50685060c51e1');
  assert.equal(checksum, recording.samplesFloat64Sha256);
});

test('display gain is constant and geometry uses actual recorded lag coordinates', () => {
  const peak = Math.max(...samples.map(Math.abs));
  for (let n = 0; n < samples.length; n++) assert.equal(signal[n], samples[n] / peak);
  for (let n = 2 * lag; n < signal.length; n++) {
    for (let axis = 0; axis < 3; axis++) {
      assert.equal(points[n * 3 + axis], Math.fround(signal[n - axis * lag] * POINT_SCALE));
    }
  }
});

test('five-second playback includes every sample and wraps without skipping the beginning', () => {
  let frame = 0;
  const visited = new Set([frame]);
  for (let n = 1; n < samples.length; n++) {
    frame = advanceFrame(frame, 1, samples.length);
    visited.add(frame);
  }
  assert.equal(visited.size, 2500);
  assert.equal(advanceFrame(frame, 1, samples.length), 0);
  assert.equal(advanceFrame(2499, 8, samples.length), 7);
});

test('waveform does not join the recording end to its beginning', () => {
  for (const frame of [0, 1, 19, 499, 500, 2499]) {
    const path = waveformPath(signal, frame);
    assert.equal(path.match(/[ML]/g).length, Math.min(frame + 1, SIGNAL_WINDOW));
    assert.ok(!/NaN|Infinity/.test(path));
  }
  assert.equal(waveformPath(signal, 0).includes('L'), false);
});

test('invalid recordings are rejected rather than replaced with synthetic data', () => {
  assert.throws(() => createNeuralFieldModel({ ...recording, sampleRate: 100 }));
  assert.throws(() => createNeuralFieldModel({ ...recording, samples: [NaN] }));
});
