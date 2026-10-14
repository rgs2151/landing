import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const base = process.env.SITE_URL || 'http://localhost:5173';
const approved = JSON.parse(readFileSync('tests/fixtures/approved-site.json'));
const projects = ['neuraljaxlib', 'neural-field-manifold', 'robust-steering', 'pylba'];
const clips = Object.assign({}, ...projects.map(project => JSON.parse(readFileSync(`public/assets/projects/${project}/previews.json`))));
assert.equal(Object.keys(clips).length, 11);
for (const clip of Object.values(clips)) {
  assert(existsSync(`public${clip.src}`));
  assert(existsSync(`public${clip.poster}`));
  for (const source of clip.sources) assert.equal(createHash('sha256').update(readFileSync(source.path)).digest('hex'), source.sha256, source.path);
}
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  async function ready() {
    await page.waitForFunction(() => document.querySelectorAll('.landing-video video[data-ready="true"]').length === 11);
    await page.waitForTimeout(300);
  }
  for (const width of [1024, 1440, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto(base); await ready();
    const layout = await page.locator('.pb-fragment').evaluateAll(nodes => nodes.map(node => ({
      id: node.dataset.piece, width: node.style.width, height: node.offsetHeight, transform: node.style.transform,
      angle: node.style.getPropertyValue('--pb-angle'), opacity: getComputedStyle(node).opacity, zIndex: getComputedStyle(node).zIndex,
    })));
    assert.deepEqual(layout, approved.splash.pieces, `Approved geometry preserved at ${width}px`);
    if (width === 1440) {
      const fonts = await page.evaluate(() => ({
        title: { text: document.querySelector('.pb-title').textContent, font: getComputedStyle(document.querySelector('.pb-title')).font },
        navFont: getComputedStyle(document.querySelector('.pb-navigation')).font,
        bodyLineHeight: getComputedStyle(document.body).lineHeight,
      }));
      const { pieces, ...expected } = approved.splash;
      assert.deepEqual(fonts, expected);
    }
  }
  assert.equal(await page.locator('.pb-fragment svg').count(), 0, 'No live SVG previews');
  const pixels = await page.locator('.landing-video video').evaluateAll(videos => videos.map(video => {
    const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d'); ctx.drawImage(video, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let visible = 0, transparent = 0, titlePixels = 0;
    for (let i = 3; i < data.length; i += 4) { if (data[i]) visible++; else transparent++; }
    for (let i = 3; i < canvas.width * 20 * 4; i += 4) if (data[i]) titlePixels++;
    return { id: video.parentElement.dataset.preview, visible, transparent, titlePixels, paused: video.paused };
  }));
  for (const row of pixels) assert(row.visible && row.transparent && row.paused, JSON.stringify(row));
  for (const row of pixels.filter(row => row.id.startsWith('neuraljax-'))) {
    assert.equal(row.titlePixels, 0, `${row.id}: heading band is transparent`);
    assert.equal(clips[row.id].hideTitle, true);
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  const videos = page.locator('.landing-video video');
  await page.waitForFunction(() => [...document.querySelectorAll('.landing-video video')].every(video => video.currentTime > .1));
  await page.getByRole('button', { name: 'Pause composition', exact: true }).click();
  await page.waitForTimeout(100);
  const paused = await videos.evaluateAll(nodes => nodes.map(video => video.currentTime));
  await page.waitForTimeout(250);
  assert.deepEqual(await videos.evaluateAll(nodes => nodes.map(video => video.currentTime)), paused);
  await page.getByRole('button', { name: 'Replay composition', exact: true }).click(); await ready();
  assert((await videos.evaluateAll(nodes => nodes.map(video => video.currentTime))).every(time => time === 0));
  await page.locator('[data-preview="landscape"] video').evaluate(video => video.dispatchEvent(new Event('ended')));
  assert.equal(await page.locator('[data-preview="landscape"] video').evaluate(video => video.currentTime), clips.landscape.loopStart / 1000);
  await page.locator('[data-menu="packages"]').hover();
  await page.locator('#pb-packages-menu a[href="#neuraljaxlib"]').click();
  await page.waitForSelector('.pb-tool-gallery .svg-asset[data-ready="true"]');
  await page.waitForFunction(() => document.querySelectorAll('.pb-frozen').length === 15);
  assert.equal(await page.locator('.pb-fragment video').count(), 0, 'Folded previews have no video decoders');
  assert.equal(await page.locator('.pb-frozen .landing-video > img').count(), 11, 'One snapshot per video, no duplicate posters');
  assert.equal(await page.locator('.pb-tool-gallery .landing-video').count(), 0, 'Detail gallery retains live sources');
  await page.getByRole('button', { name: 'Return to composition', exact: true }).click(); await ready();
  for (const id of ['cooperation', 'robust-steering', 'neural-field-manifold', 'pylba', 'neuraljaxlib', 'about']) {
    await page.goto(`${base}/#${id}`);
    await page.waitForSelector('.pb-reading h2');
    const actual = await page.evaluate(() => {
      const node = document.querySelector('.pb-reading');
      return {
        heading: node.querySelector('h2').textContent, authors: node.querySelector('.pb-authors')?.textContent,
        abstract: [...node.querySelectorAll('.pb-blurb p,.pb-about p')].map(n => n.textContent),
        font: getComputedStyle(node.querySelector('p')).font,
        links: [...node.querySelectorAll('a:not([rel="license"])')].map(n => ({ text: n.textContent, href: n.getAttribute('href') })),
      };
    });
    assert.deepEqual(JSON.parse(JSON.stringify(actual)), approved[id], id);
    assert.equal(await page.locator('[rel="license"]').count(), 0);
    if (id === 'neural-field-manifold') await page.waitForSelector('.nf-manifold canvas');
    if (id === 'robust-steering') assert.equal(await page.locator('.rs-animation').count(), 1);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(base);
  await page.waitForSelector('.pm-mobile');
  assert.equal(await page.locator('.pm-piece').count(), 15);
  assert.equal(await page.locator('.pm-portrait img').evaluate(img => img.complete && img.naturalWidth > 0), true);
  assert.deepEqual(errors, []);
  console.log('Passed: source hashes, 11 transparent videos, approved geometry at three sizes, reduced motion, playback, pause/replay, steering loop, frozen snapshots, all page content and typography, mobile composition. No screenshots or performance tests.');
} finally { await browser.close(); }
