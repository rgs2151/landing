import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

export function previewSources({ project, name }) {
  const files = {
    neuraljaxlib: [`src/shared/neuraljaxlib/vendor/${name}/animation.js`, `src/shared/neuraljaxlib/vendor/${name}/samples.js`, 'src/shared/SVGAnimation.jsx', 'src/shared/svg-animation.css'],
    'neural-field-manifold': ['src/shared/NeuralFieldAnimation.jsx', 'src/shared/neuralFieldModel.js', 'src/shared/neural-field-animation.css', 'public/assets/data/mouse-lfp.json'],
    'robust-steering': ['src/shared/RobustSteeringAnimation.jsx', 'src/shared/robustSteeringModel.js', 'src/shared/robust-steering-animation.css', 'design/references/robust-steering-animation/mgsm-examples.json', 'design/references/robust-steering-animation/steering-landscapes.json'],
    pylba: ['src/shared/pylba/PyLBAAnimation.jsx', 'src/shared/pylba/pylba.svg', 'src/shared/pylba/styles.css'],
  }[project];
  return files.map(path => ({ path, sha256: createHash('sha256').update(readFileSync(path)).digest('hex') }));
}
