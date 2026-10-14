import * as neuraljax from './neuraljaxlib/fragments';
import * as neuralField from './neural-field-manifold/fragments';
import * as steering from './robust-steering/fragments';
import * as pylba from './pylba/fragments';
import { timeline } from '../shared/robustSteeringModel';
import { loadNeuralFieldModel, PLAYBACK_SPEED } from '../shared/neuralFieldModel';

export async function exportCatalog() {
  const nfm = await loadNeuralFieldModel();
  const nfmDuration = (nfm.signal.length - 2 * nfm.lag) / (nfm.sampleRate * PLAYBACK_SPEED) * 1000;
  const entries = [];
  for (const name of ['cnn', 'rnn', 'autoencoder', 'gan', 'transformer']) {
    const [{ default: render }, { default: data }] = await Promise.all([
      import(`../shared/neuraljaxlib/vendor/${name}/animation.js`),
      import(`../shared/neuraljaxlib/vendor/${name}/samples.js`),
    ]);
    const host = document.createElement('div'); host.style.width = '300px';
    document.body.append(host);
    const renderer = render(host, data);
    const duration = renderer.duration * (name === 'cnn' ? data.samples.length : 1);
    renderer.destroy(); host.remove();
    entries.push({ project: 'neuraljaxlib', name, id: `neuraljax-${name}`, Component: neuraljax[name], duration, sourceWidth: 300, width: name === 'autoencoder' ? 480 : 360, fps: 30 });
  }
  for (const name of ['signal', 'autoregressive', 'manifold']) entries.push({
    project: 'neural-field-manifold', name, id: name, Component: neuralField[name], duration: nfmDuration,
    sourceWidth: name === 'signal' ? 250 : name === 'autoregressive' ? 375 : 270,
    width: name === 'signal' ? 300 : name === 'autoregressive' ? 450 : 960, fps: 15,
    ...(name === 'manifold' ? { crf: 10, props: { exportPixelRatio: 4 } } : {}),
  });
  for (const name of ['landscape', 'conversation']) entries.push({
    project: 'robust-steering', name, id: name, Component: steering[name],
    duration: timeline.duration * 2, loopStart: timeline.duration, sourceWidth: name === 'landscape' ? 344 : 416,
    width: name === 'landscape' ? 416 : 512, fps: 20,
    props: { previewPadding: name === 'landscape' ? 12 : 0 },
  });
  entries.push({ project: 'pylba', name: 'pylba', id: 'pylba', Component: pylba.pylba, duration: 1000, sourceWidth: 300, width: 360, fps: 30 });
  return entries;
}
