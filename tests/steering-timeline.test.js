import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const base = new URL('../design/references/robust-steering-animation/', import.meta.url);
const examples = JSON.parse(readFileSync(new URL('mgsm-examples.json', base))).examples;
const context = { Intl };
runInNewContext(readFileSync(new URL('steering-timeline.js', base), 'utf8'), context);
const timeline = context.SteeringTimeline.create(examples);

test('cycles through every recorded language and transitions back to English', () => {
  let start = 0;
  for (const entry of timeline.entries) {
    const snapshot = timeline.sample(start + 2500);
    assert.equal(snapshot.entry.language, entry.language);
    assert.equal(snapshot.question, entry.question);
    assert.equal(snapshot.answer, entry.baseline);
    if (entry.steered) assert.equal(timeline.sample(start + 7000).answer, entry.steered);
    start += entry.duration;
  }
  const wrap = timeline.sample(timeline.duration);
  assert.equal(wrap.first, false);
  assert.equal(wrap.entry.language, 'en');
  assert.equal(wrap.previous.language, 'te');
  assert.equal(wrap.question, examples.at(-1).question);
  assert.equal(wrap.answer, examples.at(-1).steered);
  assert.equal(wrap.swipe, 0);
});

test('asks and answers incrementally, erasing before retyping without a blank reset', () => {
  const start = timeline.entries[0].duration;
  assert(timeline.sample(500).question.length < examples[0].question.length);
  assert(timeline.sample(1600).answer.length < examples[0].baseline.length);
  const erase = timeline.sample(start + 150);
  assert(erase.question.length > 0 && erase.question.length < examples[0].question.length);
  assert.equal(erase.answer, examples[0].baseline);
  const typing = timeline.sample(start + 800);
  assert(typing.question.length > 0 && typing.question.length < examples[1].question.length);
  assert.equal(typing.answer, examples[0].baseline);
  for (let time = 1; time < timeline.duration * 2; time += 13) {
    const snapshot = timeline.sample(time);
    if (!snapshot.first) assert(snapshot.question || snapshot.answer);
    assert(snapshot.corruption === 0 || snapshot.phase === 'steering' || snapshot.phase === 'steered');
  }
});

test('preserves grapheme clusters for Telugu while typing and erasing', () => {
  const entry = timeline.entries.at(-1);
  const start = timeline.duration - entry.duration;
  const prefixes = new Set(entry.units.question.map((_, i) => entry.units.question.slice(0, i + 1).join('')));
  prefixes.add('');
  for (let time = 300; time < 1400; time += 11) assert(prefixes.has(timeline.sample(start + time).question));
});

test('cached display answers preserve the recorded excerpts without paraphrasing', () => {
  for (const example of examples) {
    assert.equal(example.baseline, example.source.baselineExcerpt.replaceAll('**', ''));
    if (example.steered) assert.equal(example.steered, example.source.steeredExcerpt.replaceAll('**', ''));
    assert.match(example.source.baselineSha256, /^[a-f0-9]{64}$/);
  }
});
