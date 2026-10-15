import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const clips = {};
for (const project of ['neuraljaxlib', 'neural-field-manifold', 'robust-steering', 'pylba']) {
  const folder = `public/assets/projects/${project}`;
  const source = JSON.parse(readFileSync(`${folder}/previews.json`));
  const mobile = JSON.parse(readFileSync(`${folder}/mobile-previews.json`));
  for (const [id, clip] of Object.entries(mobile)) {
    assert.equal(createHash('sha256').update(readFileSync(`public${clip.source}`)).digest('hex'), clip.sourceSha256);
    const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', `public${clip.src}`]));
    assert.equal(probe.streams[0].codec_name, 'h264');
    assert.equal(probe.streams[0].width, clip.width * 2);
    assert.equal(probe.streams[0].height, clip.height);
    assert(Math.abs(Number(probe.format.duration) * 1000 - source[id].duration) < 100, `${id}: duration preserved`);
    clips[id] = clip;
  }
}
const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  // Exercise the iPhone fallback without claiming that Chromium is an iPhone:
  // simulate a WebM decoder returning opaque pixels from the alpha capability test.
  await page.addInitScript(() => {
    const original = CanvasRenderingContext2D.prototype.getImageData;
    CanvasRenderingContext2D.prototype.getImageData = function (...args) {
      const image = original.apply(this, args);
      if (this.canvas.width === 2 && this.canvas.height === 2) for (let i = 3; i < image.data.length; i += 4) image.data[i] = 255;
      return image;
    };
  });
  await page.goto(process.env.SITE_URL || 'http://localhost:5173');
  for (const id of Object.keys(clips)) {
    await page.locator(`.pm-piece[data-piece="${id}"]`).scrollIntoViewIfNeeded();
    const player = page.locator(`.pm-packed-video[data-preview="${id}"]`);
    await player.locator('canvas[data-ready="true"]').waitFor();
    const first = await player.locator('canvas').getAttribute('data-time');
    await page.waitForFunction(({ id, first }) => {
      const time = document.querySelector(`.pm-packed-video[data-preview="${id}"] canvas`)?.dataset.time;
      return time && time !== first;
    }, { id, first });
    const pixels = await player.locator('canvas').evaluate(canvas => {
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let opaque = 0, transparent = 0;
      for (let i = 3; i < data.length; i += 4) { if (data[i] > 200) opaque++; if (data[i] === 0) transparent++; }
      return { opaque, transparent };
    });
    assert(pixels.opaque > 0 && pixels.transparent > 0, `${id}: visible artwork and restored alpha`);
  }
  await page.getByRole('button', { name: 'Pause composition', exact: true }).tap();
  await page.waitForTimeout(100);
  const paused = await page.locator('.pm-packed-video video').evaluateAll(nodes => nodes.map(video => video.currentTime));
  await page.waitForTimeout(200);
  assert.deepEqual(await page.locator('.pm-packed-video video').evaluateAll(nodes => nodes.map(video => video.currentTime)), paused);
  await page.getByRole('button', { name: 'Replay composition', exact: true }).tap();
  await page.waitForTimeout(150);
  assert(await page.locator('.pm-packed-video video').evaluateAll(nodes => nodes.every(video => video.currentTime === 0)));
  assert.deepEqual(errors, []);
  console.log('Passed: 11 H.264 compatibility videos, source hashes, dimensions/durations, decoded alpha, progressing frames, pause/replay. No screenshots; simulated alpha-limited phone decoder.');
} finally { await browser.close(); }
