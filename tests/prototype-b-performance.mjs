import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';
const browser = await playwright.chromium.launch({ executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const session = await page.context().newCDPSession(page);
  await session.send('Performance.enable');
  await page.goto('http://localhost:5173/prototype/b');
  await page.waitForTimeout(2500);
  const hasVideos = await page.locator('.pb-fragment video').count();
  for (const mode of hasVideos ? ['live', 'one-video', 'videos-paused', 'all-paused'] : ['live', 'all-paused']) {
    if (mode === 'one-video') await page.locator('.pb-fragment video').evaluateAll(videos => videos.slice(1).forEach(video => video.pause()));
    if (mode === 'videos-paused') await page.locator('.pb-fragment video').evaluateAll(videos => videos.forEach(video => video.pause()));
    if (mode === 'all-paused') await page.getByRole('button', { name: 'Pause composition', exact: true }).click();
    const before = Object.fromEntries((await session.send('Performance.getMetrics')).metrics.map(metric => [metric.name, metric.value]));
    const result = await page.evaluate(() => new Promise(resolve => {
      const deltas = []; let start, previous;
      const sample = now => {
        if (start === undefined) start = now;
        if (previous !== undefined) deltas.push(now - previous);
        previous = now;
        if (now - start < 4000) return requestAnimationFrame(sample);
        deltas.sort((a,b) => a-b);
        resolve({ fps: +(deltas.length * 1000 / (now-start)).toFixed(1), p95ms: +deltas[Math.floor(deltas.length * .95)].toFixed(1), framesOver33ms: deltas.filter(ms => ms > 33.4).length, videos: [...document.querySelectorAll('.pb-fragment video')].map(video => ({ width: video.videoWidth, height: video.videoHeight, dropped: video.getVideoPlaybackQuality().droppedVideoFrames, total: video.getVideoPlaybackQuality().totalVideoFrames })) });
      };
      requestAnimationFrame(sample);
    }));
    const after = Object.fromEntries((await session.send('Performance.getMetrics')).metrics.map(metric => [metric.name, metric.value]));
    console.log(JSON.stringify({mode,...result, mainThreadMs: Math.round((after.TaskDuration-before.TaskDuration)*1000), scriptMs: Math.round((after.ScriptDuration-before.ScriptDuration)*1000)}));
  }
} finally { await browser.close(); }
