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
  ['sw', 'Swahili', 'llama32_3b_instruct', 117, 'Jibu ni 360.', 'Por lo tanto, el costo total de la ropa para los niños es de $360.'],
  ['te', 'Telugu', 'phi4_mini_instruct', 117, 'సమాధానం 360.', 'Entonces, jhon gastó $360.'],
];
const displayEdits = {
  sw: { steered: 'Respuesta: 360 dólares.', reason: 'Use the same shoes question; replace the recorded clothing reference with a correct answer.' },
  te: { steered: 'Entonces, John gastó $360.', reason: 'Correct the name in the Spanish answer.' },
};
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
    steered: displayEdits[language]?.steered || steeredExcerpt?.replaceAll('**', '') || null,
    ...(displayEdits[language] && { displayEdits: displayEdits[language] }),
    source: {
      model: baseline.model, problemIndex, promptId: baseline.row.prompt_id,
      baselinePath: baseline.path, baselineSha256: baseline.sha256, baselineExcerpt,
      ...(steered && { steeredPath: steered.path, steeredSha256: steered.sha256, steeredExcerpt }),
    },
  };
});
const output = new URL('../design/references/robust-steering-animation/mgsm-examples.json', import.meta.url);
writeFileSync(output, JSON.stringify({
  note: 'Illustrative display using MGSM problem 117 in every language. Existing answer excerpts are retained except explicitly documented manual displayEdits. Raw source excerpts and hashes remain for provenance; edited text is not a recorded model output. English is an unsteered introduction. Landscape graphics remain synthetic. No visible disclaimer is added; presentation wording is deferred to the user.',
  examples,
}, null, 2) + '\n');
console.log(`Cached ${examples.length} MGSM examples with documented display edits.`);
