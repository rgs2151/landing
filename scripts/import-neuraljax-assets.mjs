import { readFileSync, writeFileSync, mkdirSync, copyFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';

// Explicit import only: never silently refresh approved animations during builds.
const source = resolve(process.argv[2] || '../NeuralJAXwork');
const destination = resolve('src/shared/neuraljaxlib');
const families = { cnn: 'CNN', rnn: 'RNN', autoencoder: 'Autoencoder', gan: 'GAN', transformer: 'Transformer' };
const hashes = {};
for (const [id, name] of Object.entries(families)) {
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
    hashes[`${id}/${file}`] = {
      source: createHash('sha256').update(original).digest('hex'),
      imported: createHash('sha256').update(adapted).digest('hex'),
    };
  }
}
copyFileSync(join(source, 'logo/neuraljaxlib.svg'), join(destination, 'logo.svg'));
copyFileSync(join(source, 'LICENSE'), join(destination, 'LICENSE'));
writeFileSync(join(destination, 'provenance.json'), JSON.stringify({
  repository: 'https://github.com/rgs2151/NeuralJAXlib',
  adaptation: 'ES module export boundaries only; rendering, data, geometry and timing unchanged.',
  files: hashes,
}, null, 2) + '\n');
console.log('Imported five canonical SVG renderers, cached samples, logo and license.');
