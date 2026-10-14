import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const ids = ['cnn', 'rnn', 'autoencoder', 'gan', 'transformer'];
const labels = ['CNN', 'RNN', 'Autoencoder', 'GAN', 'Transformer'];
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const [width, height] of [[1440,900], [1024,768], [1920,1080]]) {
    await page.setViewportSize({ width, height });
    await page.goto('http://localhost:5173/prototype/d');
    await page.waitForFunction(() => document.querySelectorAll('.svg-asset[data-ready="true"]').length === 5);
    await page.waitForTimeout(300);
    const fragments = page.locator('.pb-fragment[data-project="neuraljaxlib"]');
    assert.equal(await fragments.count(), 5);
    assert.equal(await fragments.locator('iframe,video,canvas,button').count(), 0);
    for (const id of ids) {
      const fragment = page.locator(`[data-piece="neuraljax-${id}"]`);
      assert.equal(await fragment.evaluate(n => getComputedStyle(n).opacity), '0.5');
      const box = await fragment.locator('.pb-orbit').boundingBox();
      assert(box.width > 60 && box.height > 60, `${id}: nonzero size`);
      assert(box.x >= 0 && box.y >= 0 && box.x + box.width <= width && box.y + box.height <= height, `${id}: on screen at ${width}`);
      assert(await fragment.locator('svg').first().evaluate(n => n.querySelectorAll('path,rect,circle,text').length > 5));
      assert(await fragment.locator('.pb-orbit').evaluate(n => {
        const box = n.getBoundingClientRect();
        return document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2)?.closest('.pb-fragment') === n.closest('.pb-fragment');
      }), `${id}: center is reachable without another fragment intercepting hover`);
    }
    assert.equal(await page.locator('[data-piece="neuraljax-rnn"] svg text').first().getAttribute('font-size'), '15.24', 'RNN label scales with the artwork, not inversely with the preview');
    const rectangles = await fragments.locator('.pb-tilt').evaluateAll(nodes => nodes.map(n => n.getBoundingClientRect().toJSON()));
    for (let a = 0; a < rectangles.length; a++) for (let b = a + 1; b < rectangles.length; b++) {
      const first = rectangles[a], second = rectangles[b];
      assert(first.right <= second.left || second.right <= first.left || first.bottom <= second.top || second.bottom <= first.top, 'New fragments do not overlap one another');
    }
    await page.locator('[data-piece="neuraljax-cnn"] .pb-orbit').hover();
    await page.waitForTimeout(450);
    assert(await fragments.evaluateAll(nodes => nodes.every(n => getComputedStyle(n).opacity === '1')));
    await page.mouse.move(10,10);
    await page.waitForTimeout(450);
    const motion = await fragments.locator('.svg-asset').evaluateAll(nodes => nodes.map(n => Number(n.dataset.elapsed)));
    await page.waitForTimeout(200);
    const advanced = await fragments.locator('.svg-asset').evaluateAll(nodes => nodes.map(n => Number(n.dataset.elapsed)));
    assert(advanced.every((value, i) => value > motion[i]), 'All five clocks advance');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  }

  await page.locator('[data-menu="packages"]').hover();
  const link = page.locator('#pb-packages-menu a[href="#neuraljaxlib"]');
  assert.equal(await link.locator('small').textContent(), '2024');
  assert(await link.locator('img').evaluate(n => n.complete && n.naturalWidth > 0));
  await link.click();
  await page.waitForSelector('.pb-tool-gallery .svg-asset[data-ready="true"]');
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pb-tool h2').textContent(), 'NeuralJAXlib');
  assert.equal(await page.locator('.pb-tool .pb-authors').textContent(), 'Rudramani Singha');
  assert.deepEqual(await page.locator('.pb-tool .pb-project-links a').allTextContents(), [' code', 'Documentation']);
  assert(await page.locator('.pb-tool h2 img').evaluate(n => n.complete && n.naturalWidth > 0 && Math.abs(n.getBoundingClientRect().height - parseFloat(getComputedStyle(n.parentElement).fontSize)) < 1));
  assert.equal(await page.locator('.pb-tool-body').count(), 5);
  assert.equal(await page.locator('.pb-fragment .svg-asset[data-playing="true"]').count(), 0, 'Folded previews are inert');
  assert.equal(await page.locator('.pb-tool-body .pb-tool-media').count(), 0, 'Figures are no longer repeated within paragraphs');
  for (const [width, height] of [[1024,768], [1440,900], [1920,1080]]) {
    await page.setViewportSize({ width, height });
    assert(await page.locator('.pb-tool-gallery').evaluate(node => {
      const row = node.querySelector('.svg-asset-mount > svg');
      const plots = [...row.children].filter(child => child.tagName.toLowerCase() === 'svg');
      const box = node.getBoundingClientRect(), rowBox = row.getBoundingClientRect();
      return plots.length === 5 && plots.every((plot, i) => Number(plot.getAttribute('y')) === 24 && (!i || Number(plot.getAttribute('x')) >= Number(plots[i-1].getAttribute('x')) + Number(plots[i-1].getAttribute('width'))))
        && rowBox.right <= box.right + 1
        && node.previousElementSibling.classList.contains('pb-project-links');
    }), `Five-panel row immediately below links at ${width}px`);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  }
  {
    const asset = page.locator('.pb-tool-gallery .svg-asset');
    await asset.getByRole('button', { name: 'Pause NeuralJAXlib animation', exact: true }).click();
    const markup = await asset.locator('.svg-asset-mount').innerHTML();
    await page.waitForTimeout(90);
    assert.equal(await asset.locator('.svg-asset-mount').innerHTML(), markup);
    await asset.getByRole('button', { name: 'Replay NeuralJAXlib animation', exact: true }).click();
    assert.equal(await asset.getAttribute('data-elapsed'), '0');
    await asset.getByRole('button', { name: 'Play NeuralJAXlib animation', exact: true }).click();
    await page.waitForTimeout(90);
    assert(Number(await asset.getAttribute('data-elapsed')) > 0);
  }
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.waitForTimeout(100);
  assert.equal(await page.locator('.pb-tool .svg-asset[data-playing="true"]').count(), 0);
  const last = page.locator('.pb-tool-gallery .svg-asset');
  await last.getByRole('button', { name: 'Play NeuralJAXlib animation', exact: true }).click();
  await page.waitForTimeout(100);
  assert.equal(await last.getAttribute('data-playing'), 'true', 'Explicit play overrides reduced motion');
  await page.emulateMedia({ reducedMotion: 'no-preference' });

  // Compare complete SVG output with the original sibling renderers at fixed times.
  const provenance = JSON.parse(readFileSync(new URL('../src/shared/neuraljaxlib/provenance.json', import.meta.url)));
  const parityIds = [...ids, 'composite'];
  for (let index = 0; index < parityIds.length; index++) {
    const id = parityIds[index];
    for (const file of id === 'composite' ? ['animation.js'] : ['animation.js', 'samples.js']) {
      const original = readFileSync(`/home/dev/NeuralJAXwork/gallery/${id}/${file}`, 'utf8');
      assert.equal(createHash('sha256').update(original).digest('hex'), provenance.files[`${id}/${file}`].source);
      await page.addScriptTag({ content: original });
    }
    assert(await page.evaluate(async ({ id, name }) => {
      const { default: React } = await import('/node_modules/.vite/deps/react.js');
      const { default: { createRoot } } = await import('/node_modules/.vite/deps/react-dom_client.js');
      const { neuraljaxAnimations, NeuralJAXGallery } = await import('/src/shared/neuraljaxlib/animations.jsx');
      const sampleNames = { cnn: 'CNN_SAMPLES', rnn: 'RNN_SAMPLES', autoencoder: 'AE_SAMPLES', gan: 'GAN_SAMPLES', transformer: 'TRANSFORMER_SAMPLES' };
      const host = document.createElement('div'), originalHost = document.createElement('div');
      for (const node of [host, originalHost]) { node.style.cssText = 'position:fixed;left:0;top:0;width:300px;visibility:hidden'; document.body.append(node); }
      const root = createRoot(host);
      const controller = window[`mount${name}`](originalHost, window[sampleNames[id]]);
      controller.pause();
      const Animation = id === 'composite' ? NeuralJAXGallery : neuraljaxAnimations.find(asset => asset.key === id).Animation;
      const normalized = node => {
        const svg = node.querySelector('svg').cloneNode(true);
        if (svg.style.backgroundColor === 'white' || svg.style.backgroundColor === 'rgb(255, 255, 255)') svg.style.background = 'transparent';
        svg.querySelectorAll('text').forEach(text => { if (text.style.fontSize) text.removeAttribute('font-size'); });
        return svg.outerHTML;
      };
      let same = true;
      try {
        for (const time of [1150, 3450, 8200]) {
          root.render(React.createElement(Animation, { compact: true, elapsedMs: time }));
          await new Promise(resolve => setTimeout(resolve, 120));
          controller.seek(time);
          same &&= normalized(host) === normalized(originalHost);
        }
      } finally { root.unmount(); controller.destroy(); host.remove(); originalHost.remove(); }
      return same;
    }, { id, name: labels[index] || 'Gallery' }), `${id}: original SVG parity at three timestamps`);
  }
  await page.goto('http://localhost:5173/prototype/d#neuraljaxlib');
  await page.waitForSelector('.pb-tool h2');
  assert.equal(await page.locator('.pb-tool h2').textContent(), 'NeuralJAXlib', 'Direct hash opens tool');
  assert.deepEqual(errors, []);
  console.log('Passed: five SVGs, three viewport sizes, opacity, live clocks, logo/year, shared details, pause/replay, reduced motion, inert stack and original-source SVG parity. No screenshots.');
} finally { await browser.close(); }
