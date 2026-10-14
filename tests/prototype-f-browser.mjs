import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const clips = JSON.parse(readFileSync('public/assets/images/previews/neuraljaxlib/clips.json'));
for (const clip of Object.values(clips)) for (const source of clip.sources) {
  assert.equal(createHash('sha256').update(readFileSync(source.path)).digest('hex'), source.sha256, 'Cached source is not stale');
}
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
const results = { geometry: [], pixels: [] };
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  async function ready() {
    await page.waitForFunction(() => document.querySelectorAll('.svg-asset[data-ready="true"]').length === 5);
    await page.waitForSelector('.nf-animation[data-frame]');
    await page.waitForTimeout(300);
  }
  for (const width of [1024, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    const layouts = [];
    for (const variant of ['e', 'f']) {
      await page.goto(`http://localhost:5173/prototype/${variant}`); await ready();
      layouts.push(await page.locator('.pb-fragment').evaluateAll(nodes => nodes.map(node => ({
        id: node.dataset.piece, transform: node.style.transform, width: node.offsetWidth, height: node.offsetHeight,
        tilt: node.style.getPropertyValue('--pb-angle'), opacity: getComputedStyle(node).opacity,
        heading: node.querySelector('.svg-asset-mount > div')?.textContent,
      }))));
    }
    assert.deepEqual(layouts[1], layouts[0], `All preview positions, dimensions, tilts and titles match at ${width}px`);
    results.geometry.push({ width, matchingPieces: layouts[0].length });
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('http://localhost:5173/prototype/f'); await ready();
  assert.equal(await page.locator('.pb-fragment [data-cached="true"]').count(), 2);
  const cached = page.locator('.pb-fragment [data-cached="true"]');
  const first = await cached.evaluateAll(nodes => nodes.map(n => n.dataset.frame));
  await page.waitForTimeout(650);
  assert.notDeepEqual(await cached.evaluateAll(nodes => nodes.map(n => n.dataset.frame)), first);
  await page.getByRole('button', { name: 'Pause composition', exact: true }).click();
  const paused = await cached.evaluateAll(nodes => nodes.map(n => n.dataset.frame));
  await page.waitForTimeout(400);
  assert.deepEqual(await cached.evaluateAll(nodes => nodes.map(n => n.dataset.frame)), paused);
  await page.getByRole('button', { name: 'Replay composition', exact: true }).click();
  await ready();
  assert.deepEqual(await cached.evaluateAll(nodes => nodes.map(n => n.dataset.frame)), ['0', '0']);
  await page.locator('[data-menu="packages"]').hover();
  await page.locator('#pb-packages-menu a[href="#neuraljaxlib"]').click();
  await page.waitForSelector('.pb-tool-gallery .svg-asset[data-ready="true"]');
  await page.waitForTimeout(900);
  assert.equal(await page.locator('.pb-tool-gallery canvas').count(), 0, 'Full gallery retains its original SVG renderers');
  assert.equal(await page.locator('.pb-tool-gallery [data-cached]').count(), 0);
  assert.equal(await page.locator('.pb-frozen').count(), 15);
  await page.getByRole('button', { name: 'Return to composition', exact: true }).click(); await ready();
  assert.equal(await cached.count(), 2);
  assert.equal(await page.locator('[data-cache-failed]').count(), 0);

  // Compare cache pixels with the actual source renderer at matching timestamps.
  // These are isolated asset rasterizations, never page screenshots.
  for (const [id, clip] of Object.entries(clips)) {
    const comparisons = await page.evaluate(async ({ id, clip }) => {
      const [{ default: render }, { default: data }, reactModule, domModule, { cachedNeuraljaxPreviews }] = await Promise.all([
        import(`/src/shared/neuraljaxlib/vendor/${id}/animation.js`), import(`/src/shared/neuraljaxlib/vendor/${id}/samples.js`),
        import('/node_modules/.vite/deps/react.js'), import('/node_modules/.vite/deps/react-dom_client.js'),
        import('/src/shared/neuraljaxlib/CachedPreviews.jsx'),
      ]);
      const React = reactModule.default || reactModule;
      const { createRoot } = domModule.default || domModule;
      const root = document.createElement('div'); root.style.cssText = 'position:fixed;left:-10000px;width:300px;color:#111'; document.body.append(root);
      const native = document.createElement('div'), target = document.createElement('div'); root.append(native, target);
      const control = render(native, data); control.pause();
      const svg = native.querySelector('svg');
      if (getComputedStyle(svg).backgroundColor === 'rgb(255, 255, 255)') svg.style.background = 'transparent';
      const reactRoot = createRoot(target);
      const rows = [];
      const frameNumbers = [0, 17, Math.floor(clip.count / 3), Math.floor(clip.count * .6), clip.count - 1];
      try {
        for (const frame of frameNumbers) {
          const ms = frame * clip.duration / clip.count;
          control.seek(ms);
          reactRoot.render(React.createElement(cachedNeuraljaxPreviews[`neuraljax-${id}`], { compact: true, elapsedMs: ms + .00001 }));
          const unique = clip.timeline[frame];
          const expectedFrame = clip.timeline.indexOf(unique);
          const start = performance.now();
          while (!target.querySelector('canvas[data-ready="true"]') || ![frame, expectedFrame].includes(Number(target.querySelector('[data-cached]')?.dataset.frame))) {
            if (performance.now() - start > 10000) throw new Error(`Frame timeout ${id}/${frame}`);
            await new Promise(resolve => setTimeout(resolve, 20));
          }
          const copy = svg.cloneNode(true);
          copy.setAttribute('width', clip.width); copy.setAttribute('height', clip.height);
          copy.style.width = `${clip.width}px`; copy.style.height = `${clip.height}px`; copy.style.color = '#111';
          const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)], { type: 'image/svg+xml' }));
          const img = new Image(); img.src = url; await img.decode();
          const canvas = document.createElement('canvas'); canvas.width = clip.width; canvas.height = clip.height;
          const ctx = canvas.getContext('2d'); ctx.drawImage(img, 0, 0); URL.revokeObjectURL(url);
          const source = ctx.getImageData(0, 0, clip.width, clip.height).data;
          const actual = target.querySelector('canvas').getContext('2d').getImageData(0, 0, clip.width, clip.height).data;
          let max = 0, total = 0, changed = 0, transparent = 0;
          for (let i = 0; i < source.length; i += 4) {
            if (source[i + 3] === 0) transparent++;
            // Compare visible colors composited over white; transparent RGB is undefined.
            for (let c = 0; c < 3; c++) {
              const a = Math.round(source[i + c] * source[i + 3] / 255 + 255 - source[i + 3]);
              const b = Math.round(actual[i + c] * actual[i + 3] / 255 + 255 - actual[i + 3]);
              const delta = Math.abs(a - b); max = Math.max(max, delta); total += delta; if (delta) changed++;
            }
            if (source[i + 3] !== actual[i + 3]) throw new Error('Alpha mismatch');
          }
          rows.push({ frame, maxChannelDifference: max, meanChannelDifference: total / (clip.width * clip.height * 3), changedChannels: changed, transparentPixels: transparent });
        }
      } finally { reactRoot.unmount(); control.destroy(); root.remove(); }
      return rows;
    }, { id, clip });
    for (const row of comparisons) {
      assert(row.maxChannelDifference <= 1 && row.meanChannelDifference < .1, `${id}: ${JSON.stringify(row)}`);
      if (id === 'autoencoder') assert(row.transparentPixels > 0);
    }
    results.pixels.push({ id, comparisons });
  }
  assert.deepEqual(errors, []);
  writeFileSync('reports/prototype-f-parity.json', JSON.stringify(results, null, 2) + '\n');
  console.log('Passed: E/F geometry, titles, pause/replay, full gallery unchanged, folding, source-frame pixel and alpha parity. No screenshots.');
} finally { await browser.close(); }
