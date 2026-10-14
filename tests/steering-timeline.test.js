import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const base = new URL('../design/references/robust-steering-animation/', import.meta.url);
const examples = JSON.parse(readFileSync(new URL('mgsm-examples.json', base))).examples;
const context = { Intl };
runInNewContext(readFileSync(new URL('steering-timeline.js', base), 'utf8'), context);
const timeline = context.SteeringTimeline.create(examples);

test('cached steering routes follow undisplayed level sets between distant points', () => {
  const { fields } = JSON.parse(readFileSync(new URL('steering-landscapes.json', base)));
  assert.deepEqual(fields.map(field => field.language), examples.map(entry => entry.language));
  for (const field of fields) {
    assert.equal(field.paths.length, 9);
    if (field.language === 'en') { assert.equal(field.route, null); continue; }
    const { d, start, end, angle } = field.route;
    assert(Math.hypot(end[0] - start[0], end[1] - start[1]) >= 130);
    assert(Number.isFinite(angle));
    const vertices = new Set(field.paths.flatMap(path => path.d.match(/-?\d+\.\d+,-?\d+\.\d+/g) || []).map(pair => pair.split(',').map(Number).join(',')));
    const points = d.slice(1).split('L');
    assert.equal(points[0], start.join(','));
    assert.equal(points.at(-1), end.join(','));
    for (const point of points) {
      assert(!vertices.has(point));
      const [x,y] = point.split(',').map(Number);
      assert(x >= 44 && x <= 296 && y >= 32 && y <= 284);
    }
  }
});

test('cycles through every recorded language and transitions back to English', () => {
  let start = 0;
  for (const entry of timeline.entries) {
    const snapshot = timeline.sample(start + 2150);
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
    assert(snapshot.question || snapshot.answer);
    assert(snapshot.corruption === 0 || snapshot.phase === 'steering' || snapshot.phase === 'steered');
  }
});

test('finishes sequential question and answer typing inside the doubled swipe', () => {
  let start = timeline.entries[0].duration;
  for (const entry of [...timeline.entries.slice(1), timeline.entries[0]]) {
    const asking = timeline.sample(start + 1399);
    assert.equal(asking.answer, asking.previous.steered || asking.previous.baseline);
    assert.equal(asking.answerTyping, false);
    assert.equal(timeline.sample(start + 1400).question, entry.question);
    const ready = timeline.sample(start + 2100);
    assert.equal(ready.question, entry.question);
    assert.equal(ready.answer, entry.baseline);
    assert.equal(ready.answerTyping, false);
    assert(ready.swipe < 1);
    assert.equal(timeline.sample(start + 1100).swipe, 0.5);
    assert.equal(timeline.sample(start + 2199).corruption, 0);
    const end = timeline.sample(start + 2200);
    assert.equal(end.swipe, 1);
    if (entry.steered) {
      assert.equal(end.phase, 'steering');
      assert(timeline.sample(start + 2201).corruption > 0);
    }
    start += entry.duration;
  }
});

test('preserves grapheme clusters for Telugu while typing and erasing', () => {
  const entry = timeline.entries.at(-1);
  const start = timeline.duration - entry.duration;
  const prefixes = new Set(entry.units.question.map((_, i) => entry.units.question.slice(0, i + 1).join('')));
  prefixes.add('');
  for (let time = 300; time < 1400; time += 11) assert(prefixes.has(timeline.sample(start + time).question));
});

test('one shared problem preserves source excerpts and documents manual display edits', () => {
  for (const example of examples) {
    assert.equal(example.source.problemIndex, 117);
    assert.match(example.baseline, /360/);
    assert.equal(example.baseline, example.source.baselineExcerpt.replaceAll('**', ''));
    if (example.steered) {
      assert.match(example.steered, /360/);
      assert.equal(example.steered, example.displayEdits?.steered || example.source.steeredExcerpt.replaceAll('**', ''));
    }
    assert.match(example.source.baselineSha256, /^[a-f0-9]{64}$/);
  }
});
