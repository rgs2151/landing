// Shared by the isolated animation and its focused Node tests.
globalThis.SteeringTimeline = (() => {
  const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  const units = text => [...segmenter.segment(text || '')].map(item => item.segment);
  const clamp = value => Math.max(0, Math.min(1, value));
  const prefix = (text, amount) => text.slice(0, Math.floor(text.length * clamp(amount))).join('');

  function create(examples) {
    const entries = examples.map(example => ({ ...example, units: {
      question: units(example.question), baseline: units(example.baseline), steered: units(example.steered),
    }, duration: example.language === 'en' ? 6500 : 7200 }));
    const duration = entries.reduce((sum, entry) => sum + entry.duration, 0);
    function sample(elapsed) {
      let time = elapsed % duration;
      let index = 0;
      while (time >= entries[index].duration) time -= entries[index++].duration;
      const entry = entries[index];
      const previous = entries[(index + entries.length - 1) % entries.length];
      const first = elapsed < entries[0].duration;
      const progress = (start, span) => clamp((time - start) / span);
      const questionStart = first ? 0 : 300;
      const answerErase = first ? 1100 : 1400;
      const answerStart = answerErase + 250;
      const answerDuration = first ? 600 : 450;
      const answerEnd = answerStart + answerDuration;
      const previousAnswer = previous.steered || previous.baseline;
      let phase, question, answer;
      if (!first && time < questionStart) {
        phase = 'question-erase';
        question = prefix(previous.units.question, 1 - progress(0, questionStart));
      } else {
        question = prefix(entry.units.question, progress(questionStart, 1100));
        if (first && !question) question = entry.units.question[0];
        phase = time < questionStart + 1100 ? 'question' : 'answer-erase';
      }
      if (time < answerErase) answer = first ? '' : previousAnswer;
      else if (time < answerStart) answer = first ? '' : prefix(previous.units.steered.length ? previous.units.steered : previous.units.baseline, 1 - progress(answerErase, 250));
      else {
        answer = prefix(entry.units.baseline, progress(answerStart, answerDuration));
        phase = time < answerEnd ? 'baseline' : 'baseline-hold';
      }
      let corruption = 0;
      if (entry.steered && time >= 2200) {
        corruption = progress(2200, 2800);
        phase = corruption < 1 ? 'steering' : 'steered';
        answer = corruption < 1 ? entry.baseline : entry.steered;
      }
      return {
        index, entry, previous, first, time, phase, question, answer, corruption,
        questionTyping: time < questionStart + 1100,
        answerTyping: time >= answerErase && time < answerEnd,
        questionLanguage: !first && time < questionStart ? previous.language : entry.language,
        answerLanguage: time < answerStart && !first ? (previous.steered ? 'es' : previous.language) : phase === 'steered' ? 'es' : entry.language,
        swipe: first ? 0 : progress(0, 2200),
      };
    }
    return { entries, duration, sample, units };
  }
  return { create };
})();
