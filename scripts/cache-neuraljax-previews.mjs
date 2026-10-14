import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';
import { createRequire } from 'node:module';
const sharp = createRequire('/tmp/landing-browser-check/package.json')('sharp');

// Export only the existing SVG artwork, not page screenshots. Titles remain HTML.
const directory = 'public/assets/images/previews/neuraljaxlib';
mkdirSync(directory, { recursive: true });
const previous = existsSync(`${directory}/clips.json`) ? JSON.parse(readFileSync(`${directory}/clips.json`)) : {};
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
const manifest = {};
try {
  const page = await browser.newPage();
  await page.goto('http://localhost:5173/prototype/e');
  for (const [id, fps, width] of [['gan', 30, 57], ['autoencoder', 60, 480]]) {
    const meta = await page.evaluate(async ({ id, width }) => {
      const [{ default: render }, { default: data }] = await Promise.all([
        import(`/src/shared/neuraljaxlib/vendor/${id}/animation.js`),
        import(`/src/shared/neuraljaxlib/vendor/${id}/samples.js`),
      ]);
      const root = document.createElement('div');
      root.style.cssText = 'position:fixed;left:-10000px;width:300px;color:#111';
      document.body.append(root);
      const controller = render(root, data);
      controller.pause();
      const svg = root.querySelector('svg');
      if (getComputedStyle(svg).backgroundColor === 'rgb(255, 255, 255)') svg.style.background = 'transparent';
      const view = svg.viewBox.baseVal;
      const height = Math.round(width * view.height / view.width);
      const heading = root.firstElementChild;
      window.exportAsset = { root, controller, svg, width, height };
      return { duration: controller.duration, width, height, viewBox: svg.getAttribute('viewBox'),
        heading: heading.textContent, headingStyle: heading.getAttribute('style'),
        svgStyle: svg.getAttribute('style'), label: svg.getAttribute('aria-label') };
    }, { id, width });
    const count = Math.ceil(meta.duration * fps / 1000);
    const timeline = [], frames = [], hashes = new Map();
    for (let frame = 0; frame < count; frame++) {
      const png = await page.evaluate(async time => {
        const { controller, svg, width, height } = window.exportAsset;
        controller.seek(time);
        const copy = svg.cloneNode(true);
        copy.setAttribute('width', width); copy.setAttribute('height', height);
        copy.style.width = `${width}px`; copy.style.height = `${height}px`; copy.style.color = '#111';
        const blob = new Blob([new XMLSerializer().serializeToString(copy)], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob), image = new Image();
        image.src = url; await image.decode();
        const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
        canvas.getContext('2d').drawImage(image, 0, 0);
        URL.revokeObjectURL(url);
        return canvas.toDataURL('image/png').split(',')[1];
      }, frame * meta.duration / count);
      const buffer = Buffer.from(png, 'base64');
      const hash = createHash('sha256').update(buffer).digest('hex');
      if (!hashes.has(hash)) { hashes.set(hash, frames.length); frames.push(buffer); }
      timeline.push(hashes.get(hash));
    }
    const perSheet = id === 'gan' ? frames.length : 12;
    const columns = id === 'gan' ? 12 : 4;
    const sheets = [];
    for (let start = 0; start < frames.length; start += perSheet) {
      const group = frames.slice(start, start + perSheet);
      const width = columns * meta.width, height = Math.ceil(group.length / columns) * meta.height;
      const pixels = Buffer.alloc(width * height * 4);
      // Copy RGBA directly: alpha compositing would round translucent dot colors.
      for (let i = 0; i < group.length; i++) {
        const rgba = await sharp(group[i]).ensureAlpha().raw().toBuffer();
        for (let y = 0; y < meta.height; y++) {
          const offset = ((Math.floor(i / columns) * meta.height + y) * width + i % columns * meta.width) * 4;
          rgba.copy(pixels, offset, y * meta.width * 4, (y + 1) * meta.width * 4);
        }
      }
      const sheet = await sharp(pixels, { raw: { width, height, channels: 4 } }).webp({ lossless: true, effort: 6 }).toBuffer();
      const revision = createHash('sha256').update(sheet).digest('hex').slice(0, 12);
      const file = `${id}-${sheets.length}-${revision}.webp`;
      writeFileSync(`${directory}/${file}`, sheet);
      sheets.push({ src: `/assets/images/previews/neuraljaxlib/${file}`, start, count: group.length, columns, bytes: sheet.length });
    }
    const sources = ['animation.js', 'samples.js'].map(file => ({
      path: `src/shared/neuraljaxlib/vendor/${id}/${file}`,
      sha256: createHash('sha256').update(readFileSync(`src/shared/neuraljaxlib/vendor/${id}/${file}`)).digest('hex'),
    }));
    manifest[id] = { ...meta, fps, count, timeline, sheets, sources };
    await page.evaluate(() => { window.exportAsset.controller.destroy(); window.exportAsset.root.remove(); });
    console.log(`${id}: ${count} frames, ${frames.length} unique, ${sheets.length} sheets, ${sheets.reduce((sum, s) => sum + s.bytes, 0)} bytes`);
  }
  writeFileSync(`${directory}/clips.json`, JSON.stringify(manifest, null, 2) + '\n');
  const currentFiles = new Set(Object.values(manifest).flatMap(clip => clip.sheets.map(sheet => sheet.src)));
  for (const clip of Object.values(previous)) for (const sheet of clip.sheets) {
    if (!currentFiles.has(sheet.src)) unlinkSync(`public${sheet.src}`);
  }
} finally { await browser.close(); }
