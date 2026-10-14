import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const base = process.env.SITE_URL || 'http://localhost:5173';
const approved = JSON.parse(readFileSync('tests/fixtures/approved-site.json'));
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, reducedMotion: 'reduce' });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const overflow = async label => assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No page overflow: ${label}`);
  for (const width of [320, 390, 430, 768, 844]) {
    await page.setViewportSize({ width, height: width === 844 ? 390 : 844 });
    await page.goto(base);
    await page.waitForSelector('.pm-piece');
    assert.equal(await page.locator('.pm-piece').count(), 15);
    await overflow(`splash ${width}`);
    const fit = await page.locator('.pm-name span').evaluateAll(nodes => nodes.every(node => {
      const box = node.getBoundingClientRect(); return box.left >= 0 && box.right <= innerWidth;
    }));
    assert(fit, 'Name stays on screen');
    await page.locator('[data-menu="research"]').tap();
    await page.locator('#pb-research-menu a[href="#cooperation"]').tap();
    await page.waitForSelector('.pm-reading h2');
    assert.equal(await page.locator('.pm-reading h2').textContent(), approved.cooperation.heading);
    await overflow(`reading ${width}`);
    await page.getByRole('button', { name: 'Return to composition', exact: true }).tap();
    await page.waitForSelector('.pm-piece');
  }
  await page.setViewportSize({ width: 390, height: 844 });
  for (const id of await page.locator('.pm-piece').evaluateAll(nodes => nodes.map(node => node.dataset.piece))) {
    await page.locator(`.pm-piece[data-piece="${id}"]`).tap();
    await page.waitForSelector('.pm-reading');
    await page.getByRole('button', { name: 'Return to composition', exact: true }).tap();
  }
  for (const id of ['cooperation', 'robust-steering', 'neural-field-manifold', 'neuraljaxlib', 'pylba', 'about']) {
    await page.goto(`${base}/#${id}`);
    await page.waitForSelector('.pm-reading h2');
    await page.waitForTimeout(150);
    const content = await page.locator('.pm-reading').evaluate(node => ({
      heading: node.querySelector('h2').textContent,
      authors: node.querySelector('.pb-authors')?.textContent,
      abstract: [...node.querySelectorAll('.pb-blurb p,.pb-about p')].map(p => p.textContent),
      links: [...node.querySelectorAll('a')].map(a => ({ text: a.textContent, href: a.getAttribute('href') })),
    }));
    const { font, ...expected } = approved[id];
    assert.deepEqual(JSON.parse(JSON.stringify(content)), expected, `Frozen copy and links: ${id}`);
    assert.equal(await page.locator('.pm-piece video').count(), 0, 'Splash players unmounted on detail');
    await overflow(id);
    if (id === 'about') {
      assert.equal(await page.locator('.pb-about-photo-slot img').count(), 1);
    } else if (id !== 'pylba') {
      await page.getByRole('button', { name: 'Enlarge figures' }).tap();
      await overflow(`enlarged ${id}`);
      if (id !== 'pylba') assert(await page.locator('.pb-project-media,.pb-tool-gallery').evaluate(node => node.scrollWidth > node.clientWidth), 'Wide figures can be panned');
      await page.getByRole('button', { name: 'Fit figures to screen' }).tap();
    }
    if (id === 'neural-field-manifold') {
      await page.waitForSelector('.nf-manifold canvas');
      assert(await page.locator('.nf-manifold canvas').evaluate(canvas => canvas.width > 0 && canvas.height > 0));
    }
  }
  await page.goto(base);
  await page.locator('[data-menu="packages"]').tap();
  await page.locator('#pb-packages-menu a[href="#neuraljaxlib"]').tap();
  await page.waitForSelector('.pb-tool-gallery .svg-asset[data-ready="true"]');
  await page.goBack();
  await page.waitForSelector('.pm-piece');
  await page.goForward();
  await page.waitForSelector('.pm-reading');
  await page.getByRole('button', { name: 'Return to composition', exact: true }).tap();
  await page.locator('.pm-piece[data-piece="neuraljax-gan"]').tap();
  await page.waitForSelector('.pm-reading');
  await page.getByRole('button', { name: 'Return to composition', exact: true }).tap();
  assert(await page.evaluate(() => scrollY > 300), 'Return restores collage scroll');
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.locator('.pm-piece[data-piece="neuraljax-gan"] video').waitFor();
  await page.waitForFunction(() => document.querySelector('.pm-piece[data-piece="neuraljax-gan"] video')?.currentTime > .1);
  await page.getByRole('button', { name: 'Pause composition', exact: true }).tap();
  assert(await page.locator('.pm-piece video').evaluateAll(videos => videos.every(v => v.paused)));
  await page.getByRole('button', { name: 'Replay composition', exact: true }).tap();
  await page.waitForTimeout(100);
  assert(await page.locator('.pm-piece video').evaluateAll(videos => videos.every(v => v.currentTime === 0)));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForSelector('.pb-canvas');
  assert.equal(await page.locator('.pm-mobile').count(), 0, 'Desktop remains separate');
  assert.deepEqual(errors, []);
  console.log('Passed: touch navigation, all project copy/links, five widths, landscape, figure enlargement, About, browser history, scroll restoration, playback, pause/replay, desktop switch. No screenshots.');
} finally { await browser.close(); }
