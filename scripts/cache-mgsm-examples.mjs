import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';

const root = resolve(process.argv[2] || '../controller/code');
const selections = [
  ['en', 'English', 'qwen3_4b', 117, 'The answer is 360.', null],
  ['zh', 'Chinese', 'qwen3_4b', 117, '答案是 360 美元。', '**Respuesta: 360 dólares.**'],
  ['fr', 'French', 'qwen3_4b', 117, 'La réponse est **360**.', '**La respuesta es: 360.**'],
  ['ja', 'Japanese', 'qwen3_4b', 117, '答えは **360ドル** です。', '**Respuesta: 360 dólares.**'],
  ['sw', 'Swahili', 'qwen3_4b', 96, '**Jibu:** 3 saa.', '**3 horas.**'],
  ['te', 'Telugu', 'phi4_mini_instruct', 117, 'సమాధానం 360.', 'Entonces, jhon gastó $360.'],
];
const examples = selections.map(([language, label, model, problemIndex, baselineExcerpt, steeredExcerpt]) => {
  function load(method, excerpt) {
    const path = `benchmarks/mgsm/cache/${model}/evaluations/kv_cache_off/generations/mgsm_${language}/${method}/final.json`;
    const bytes = readFileSync(resolve(root, path));
    const source = JSON.parse(bytes);
    const row = source.repetitions[0].rows.find(row => row.problem_index === problemIndex);
    assert(row && row.completion.includes(excerpt), `Excerpt must occur verbatim in ${path}`);
    return { path, sha256: createHash('sha256').update(bytes).digest('hex'), model: source.identity.model, row };
  }
  const baseline = load('original', baselineExcerpt);
  const steered = steeredExcerpt ? load('h_infinity', steeredExcerpt) : null;
  if (steered) assert.equal(baseline.row.question, steered.row.question);
  return {
    language, label, question: baseline.row.question,
    baseline: baselineExcerpt.replaceAll('**', ''),
    steered: steeredExcerpt?.replaceAll('**', '') || null,
    source: {
      model: baseline.model, problemIndex, promptId: baseline.row.prompt_id,
      baselinePath: baseline.path, baselineSha256: baseline.sha256, baselineExcerpt,
      ...(steered && { steeredPath: steered.path, steeredSha256: steered.sha256, steeredExcerpt }),
    },
  };
});
const output = new URL('../design/references/robust-steering-animation/mgsm-examples.json', import.meta.url);
writeFileSync(output, JSON.stringify({
  note: 'Selected recorded MGSM questions and verbatim final-answer excerpts, with Markdown emphasis removed. These are not full completions or aggregate performance claims. English is an unsteered introduction; other rows pair original and H-infinity generations. Swahili uses a different problem. Landscape graphics remain synthetic.',
  examples,
}, null, 2) + '\n');
console.log(`Cached ${examples.length} source-verified MGSM examples.`);
