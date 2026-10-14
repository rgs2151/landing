import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const provenance = JSON.parse(readFileSync('src/shared/neuraljaxlib/brain-provenance.json'));
const abstract = readFileSync('src/content/neuraljaxlib-abstract.md', 'utf8').trim();
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://localhost:5173/prototype/f#neuraljaxlib');
  await page.waitForFunction(() => document.querySelectorAll('.pb-project .svg-asset[data-ready="true"]').length === 3);
  for (const width of [1024, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 }); await page.waitForTimeout(300);
    const layout = await page.evaluate(() => {
      const box = node => { const r = node.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom }; };
      const body = document.querySelector('.pb-illustrated-abstract');
      const p = body.querySelector('p'); const range = document.createRange(); range.selectNodeContents(p);
      return {
        text: p.textContent,
        width: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth,
        panels: [...document.querySelectorAll('[data-gallery-panel]')].map(node => ({ id: node.dataset.galleryPanel,
          heading: node.firstElementChild.textContent, headingVisibility: getComputedStyle(node.firstElementChild).visibility,
          aria: node.querySelector('svg').getAttribute('aria-label'), box: box(node.querySelector('svg')) })),
        figures: [...body.querySelectorAll('.pb-abstract-figure')].map(node => ({ side: node.dataset.side, box: box(node.querySelector('.svg-asset')) })),
        lines: [...range.getClientRects()].map(r => ({ left: r.left, right: r.right, top: r.top, bottom: r.bottom })),
        lineHeight: parseFloat(getComputedStyle(p).lineHeight), bodyTop: body.getBoundingClientRect().top,
      };
    });
    assert.equal(layout.text, abstract);
    assert(layout.scrollWidth <= layout.width, 'No horizontal overflow');
    assert.deepEqual(layout.panels.map(p => p.id), ['cnn', 'rnn', 'autoencoder', 'gan', 'transformer']);
    assert.match(layout.panels[0].aria, /brain MRI/i); assert.match(layout.panels[3].aria, /brain MRI/i);
    assert(layout.panels.every(p => p.heading.startsWith('nj.Module.') && p.headingVisibility === 'visible'));
    assert.equal(layout.figures.length, 2);
    assert(Math.abs(layout.figures[0].box.top - layout.bodyTop - 2 * layout.lineHeight) < 1, 'CNN starts on abstract line three');
    assert(Math.abs(layout.figures[1].box.top - layout.bodyTop - 5 * layout.lineHeight) < 1, 'GAN starts three lines later');
    for (const line of layout.lines) for (const figure of layout.figures) {
      const box = figure.box;
      assert(line.right <= box.left + 1 || line.left >= box.right - 1 || line.bottom <= box.top + 1 || line.top >= box.bottom - 1, 'Abstract text wraps without overlap');
    }
  }
  assert.equal(await page.locator('.pb-abstract-figure [data-asset="cnn"]').count(), 1);
  assert.equal(await page.locator('.pb-abstract-figure [data-asset="gan"]').count(), 1);

  // Imported artwork must match the audited originals, not just their dimensions.
  for (const [id, name] of [['cnn_brain_mri', 'CNN'], ['gan_brain_mri', 'GAN']]) {
    for (const file of ['animation.js', 'samples.js']) {
      const original = readFileSync(`/home/dev/NeuralJAXwork/gallery/${id}/${file}`, 'utf8');
      assert.equal(createHash('sha256').update(original).digest('hex'), provenance.files[`${id}/${file}`].source);
      await page.addScriptTag({ content: original });
    }
    assert(await page.evaluate(async ({ id, name }) => {
      const [{ default: render }, { default: samples }] = await Promise.all([
        import(`/src/shared/neuraljaxlib/vendor/${id}/animation.js`), import(`/src/shared/neuraljaxlib/vendor/${id}/samples.js`),
      ]);
      const hosts = [document.createElement('div'), document.createElement('div')];
      for (const host of hosts) { host.style.cssText = 'position:fixed;left:-10000px;width:300px'; document.body.append(host); }
      const controls = [render(hosts[0], samples), window[`mount${name}`](hosts[1], window[`${name}_SAMPLES`])];
      try {
        for (const time of [0, 1150, 3450, 14300]) {
          controls.forEach(c => c.seek(time));
          if (hosts[0].innerHTML !== hosts[1].innerHTML) return false;
        }
        return true;
      } finally { controls.forEach(c => c.destroy()); hosts.forEach(h => h.remove()); }
    }, { id, name }), `${id}: exact source-renderer parity`);
  }
  await page.goto('http://localhost:5173/prototype/e#neuraljaxlib');
  await page.waitForSelector('.pb-tool-gallery .svg-asset[data-ready="true"]');
  assert.doesNotMatch(await page.locator('[data-gallery-panel="cnn"] svg').getAttribute('aria-label'), /brain MRI/i);
  assert.equal(await page.locator('.pb-abstract-figure').count(), 0);
  assert.deepEqual(errors, []);
  console.log('Passed: audited brain-source parity, five-panel titles, legacy figures, exact abstract, line-three/six floats without text overlap at three widths, frozen E unchanged. No screenshots.');
} finally { await browser.close(); }
