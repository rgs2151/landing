import assert from 'node:assert/strict';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const base = process.env.SITE_URL || 'http://localhost:5173';
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelectorAll('.landing-video video[data-ready="true"]').length === 11);
  await page.getByRole('button', { name: 'Pause composition', exact: true }).click();
  await page.locator('[data-menu="packages"]').hover();
  assert.equal(await page.locator('#pb-packages-menu a[href="#pylba"] span').textContent(), 'pyLBA: Linear Ballistic Accumulators');
  assert.equal(await page.locator('#pb-packages-menu a[href="#pylba"] small').textContent(), '2025');
  assert.equal(await page.locator('#pb-packages-menu a[href="#neuraljaxlib"] span').textContent(), 'NeuralJAXlib');
  assert.equal(await page.locator('#pb-packages-menu a[href="#neuraljaxlib"] small').textContent(), '2026');
  assert.deepEqual(await page.locator('#pb-packages-menu a').evaluateAll(nodes => nodes.map(n => n.getAttribute('href'))), ['#neuraljaxlib', '#pylba']);
  assert.equal(await page.locator('#pb-packages-menu img').count(), 0);
  await page.mouse.move(5, 5);

  for (const [project, count, piece, menu] of [
    ['neuraljaxlib', 5, 'neuraljax-rnn', 'packages'],
    ['cooperation', 4, 'nature-0', 'research'],
    ['neural-field-manifold', 3, null, 'research'],
    ['robust-steering', 2, null, 'research'],
    ['pylba', 1, null, 'packages'],
  ]) {
    // Capture the initial state in the browser, before a cold gallery mount can
    // delay the next Playwright round trip past the entire fade.
    await page.evaluate(() => {
      window.arrivalStartOpacity = null;
      const observer = new MutationObserver(() => {
        const nodes = [...document.querySelectorAll('.pb-fragment[data-arriving="true"]')];
        if (!nodes.length) return;
        window.arrivalStartOpacity = nodes.map(node => Number(getComputedStyle(node).opacity));
        observer.disconnect();
      });
      observer.observe(document.querySelector('.pb-canvas'), { attributes: true, subtree: true });
    });
    if (piece) await page.locator(`[data-piece="${piece}"] .pb-orbit`).click();
    else {
      await page.locator(`[data-menu="${menu}"]`).hover();
      await page.locator(`#pb-${menu}-menu a[href="#${project}"]`).click();
    }
    const arrivals = page.locator('.pb-fragment[data-arriving="true"]');
    await page.waitForSelector('.pb-fragment[data-arriving="true"]');
    assert.equal(await arrivals.count(), count);
    assert(await arrivals.evaluateAll(nodes => nodes.every(n => n.dataset.project === document.querySelector('main').dataset.selected)));
    assert(await page.evaluate(() => window.arrivalStartOpacity?.every(opacity => opacity > .5)), 'Travel stays visible before its delayed fade');
    assert(await arrivals.locator('.pb-orbit').evaluateAll(nodes => nodes.every(n => getComputedStyle(n).pointerEvents === 'none')));
    await page.waitForTimeout(1000);
    const landing = await page.evaluate(() => {
      const reading = document.querySelector('.pb-reading');
      const media = reading.querySelector('.pb-project-media, .pb-tool-gallery, .pb-tool-media') || reading;
      const box = media.getBoundingClientRect();
      const target = { x: box.left + box.width / 2, y: Math.min(innerHeight - 80, Math.max(120, box.top + box.height / 2)) };
      const portrait = document.querySelector('.pc-scene-portrait[data-location="stack"]').getBoundingClientRect();
      const stackCenter = { x: portrait.left + portrait.width / 2, y: portrait.top + portrait.height / 2 };
      return [...document.querySelectorAll('.pb-fragment')].map(node => {
        const rect = node.getBoundingClientRect();
        return { arriving: node.dataset.arriving === 'true', x: rect.left + rect.width / 2, y: rect.top + rect.height / 2,
          opacity: Number(getComputedStyle(node).opacity), target, stackCenter };
      });
    });
    for (const item of landing) {
      if (item.arriving) {
        assert.equal(item.opacity, 0);
        assert(Math.abs(item.x - item.target.x) < 3 && Math.abs(item.y - item.target.y) < 3, `${project}: selected assets reach the figure center`);
      } else {
        assert(item.x < 300 && item.opacity === 1, `${project}: other assets remain in left stack`);
        assert(Math.abs(item.x - item.stackCenter.x) < 1 && Math.abs(item.y - item.stackCenter.y) < 1, `${project}: folded cards share the photo's center regardless of size or rotation`);
      }
    }
    assert.equal(await page.locator('.pb-fragment .svg-asset[data-playing="true"]').count(), 0, 'Folded/arriving SVG previews stop rendering');
    if (project === 'neuraljaxlib') assert.equal(await page.locator('.pb-tool h2 img').count(), 1, 'Page title retains its logo');
    if (project === 'pylba') assert((await page.locator('.pb-tool h2').textContent()).startsWith('\u{1f680}'), 'Page title retains rocket');
    await page.getByRole('button', { name: 'Return to composition', exact: true }).click();
    await page.waitForSelector('.pb-canvas[data-transitioning="false"][data-open="false"]');
    await page.mouse.move(5, 5);
    await page.waitForTimeout(350);
    assert.equal(await arrivals.count(), 0);
    assert(await page.locator('.pb-fragment').evaluateAll(nodes => nodes.every(n => Number(getComputedStyle(n).opacity) >= .75)), 'All previews return from their own destinations');
  }

  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.locator('[data-menu="packages"]').hover();
  await page.locator('#pb-packages-menu a[href="#neuraljaxlib"]').click();
  await page.waitForSelector('.pb-tool-gallery [data-ready="true"]');
  assert(await page.locator('[data-arriving="true"]').evaluateAll(nodes => nodes.every(n => getComputedStyle(n).opacity === '0' && getComputedStyle(n).transitionDuration === '0s')), 'Reduced motion skips travel and fade');
  assert.deepEqual(errors, []);
  console.log('Passed: Navigation labels, page-title icons, five project arrivals, delayed fades, left stack, back navigation, reduced motion. No screenshots.');
} finally { await browser.close(); }
