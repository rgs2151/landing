import { readFileSync, writeFileSync, mkdirSync, copyFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';

const source = resolve(process.argv[2] || '../NeuralJAXwork');
const destination = resolve('src/shared/neuraljaxlib');
const files = {};
for (const [id, name] of [['cnn_brain_mri', 'CNN'], ['gan_brain_mri', 'GAN']]) {
  const folder = join(destination, 'vendor', id);
  mkdirSync(folder, { recursive: true });
  for (const file of ['animation.js', 'samples.js']) {
    const original = readFileSync(join(source, 'gallery', id, file), 'utf8');
    let adapted;
    if (file === 'animation.js') {
      adapted = original.replace(/\(function\s*\(\)\s*\{/, 'export default (function () {')
        .replace(`window.mount${name}`, 'const mount')
        .replace(/\}\)\(\);\s*$/, '  return mount;\n})();\n');
      if (adapted === original || adapted.includes(`window.mount${name}`)) throw new Error(`Unrecognized renderer: ${id}`);
    } else {
      const json = original.slice(original.indexOf('=') + 1).trim().replace(/;$/, '');
      JSON.parse(json);
      adapted = `export default ${json};\n`;
    }
    writeFileSync(join(folder, file), adapted);
    files[`${id}/${file}`] = {
      source: createHash('sha256').update(original).digest('hex'),
      imported: createHash('sha256').update(adapted).digest('hex'),
    };
  }
  copyFileSync(join(source, 'gallery', id, 'README.md'), join(folder, 'README.md'));
}
writeFileSync(join(destination, 'brain-provenance.json'), JSON.stringify({
  repository: 'https://github.com/rgs2151/NeuralJAXlib',
  adaptation: 'ES module exports only. Original MNIST/Fashion-MNIST assets retained unchanged.',
  files,
}, null, 2) + '\n');
mkdirSync(join(destination, 'vendor', 'neuroscience'), { recursive: true });
copyFileSync(join(source, 'gallery', 'neuroscience', 'README.md'), join(destination, 'vendor', 'neuroscience', 'README.md'));
console.log('Imported audited brain MRI CNN and GAN; sibling repository and existing assets untouched.');
