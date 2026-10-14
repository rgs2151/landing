import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
const samples = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const session = await page.context().newCDPSession(page);
  await session.send('Performance.enable');
  await page.goto('http://localhost:5173/prototype/e');
  await page.waitForFunction(() => document.querySelectorAll('.svg-asset[data-ready="true"]').length === 5);
  await page.waitForTimeout(1500);
  const metrics = async () => Object.fromEntries((await session.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
  async function sample(name, duration, action) {
    const before = await metrics();
    const timing = await page.evaluate(({ duration, action }) => new Promise(resolve => {
      const deltas = [], longTasks = [];
      const observer = new PerformanceObserver(list => longTasks.push(...list.getEntries().map(entry => entry.duration)));
      observer.observe({ type: 'longtask' });
      let start, previous;
      function tick(now) {
        if (start === undefined) {
          start = now;
          setTimeout(() => {
            if (action === 'back') document.querySelector('.pb-dock-back').click();
            else if (action === 'hover' || action === 'unhover') {
              document.querySelector('[data-piece="neuraljax-rnn"]').dispatchEvent(new PointerEvent(action === 'hover' ? 'pointerover' : 'pointerout', { bubbles: true }));
            } else if (action) document.querySelector(`.pb-navigation a[href="#${action}"]`).click();
          }, 50);
        }
        if (previous !== undefined) deltas.push(now - previous);
        previous = now;
        if (now - start < duration) return requestAnimationFrame(tick);
        observer.disconnect();
        deltas.sort((a, b) => a - b);
        resolve({ seconds: (now - start) / 1000, fps: deltas.length * 1000 / (now - start),
          p95ms: deltas[Math.floor(deltas.length * .95)], p99ms: deltas[Math.floor(deltas.length * .99)],
          maxFrameMs: deltas.at(-1), framesOver33ms: deltas.filter(ms => ms > 33.5).length,
          longTasks, frozen: document.querySelectorAll('.pb-frozen').length,
          activeProject: document.querySelector('main').dataset.selected,
          hover: document.querySelector('.pb-canvas').dataset.active });
      }
      requestAnimationFrame(tick);
    }), { duration, action });
    const after = await metrics();
    const row = { name, ...timing, ...Object.fromEntries(['TaskDuration', 'ScriptDuration', 'LayoutDuration', 'RecalcStyleDuration'].map(key => [key + 'MsPerSecond', (after[key] - before[key]) * 1000 / timing.seconds])) };
    samples.push(row);
    console.log(`${name}: ${timing.fps.toFixed(1)} FPS, p95 ${timing.p95ms.toFixed(1)}ms, max ${timing.maxFrameMs.toFixed(1)}ms, ${timing.longTasks.length} long tasks`);
  }
  await sample('live-30-seconds', 30000);
  await sample('hover-tools', 4000, 'hover');
  assert.equal(samples.at(-1).hover, 'neuraljaxlib');
  await sample('unhover', 2000, 'unhover');
  for (let run = 1; run <= 2; run++) {
    for (const project of ['cooperation', 'neural-field-manifold', 'robust-steering', 'neuraljaxlib', 'pylba', 'about']) {
      await sample(`open-${project}-${run}`, 1800, project);
      assert.equal(samples.at(-1).activeProject, project);
      if (run === 1) await sample(`settled-${project}`, 2000);
      if (project === 'about') {
        assert.equal(await page.locator('.pb-about p').first().textContent(), 'I am a data scientist in the Program in Memory Longevity at UT Southwestern Medical Center. I received my M.S. in Biomedical Engineering from Columbia University. My research combines computational neuroscience and machine learning to study neural dynamics and information processing in biological and artificial neural networks.');
        assert.equal(await page.locator('.pb-about p').first().locator('a').getAttribute('href'), 'https://memorylongevity.org/');
      }
      await sample(`return-${project}-${run}`, 1800, 'back');
      assert.equal(samples.at(-1).activeProject, '');
      await page.waitForTimeout(400);
    }
  }
  await session.send('Profiler.enable');
  await session.send('Profiler.start');
  await page.waitForTimeout(10000);
  const { profile } = await session.send('Profiler.stop');
  const nodes = new Map(profile.nodes.map(node => [node.id, node.callFrame]));
  const totals = new Map();
  profile.samples.forEach((id, i) => {
    const frame = nodes.get(id);
    const key = `${frame.url.replace('http://localhost:5173/', '')}:${frame.lineNumber + 1} ${frame.functionName || '(anonymous)'}`;
    totals.set(key, (totals.get(key) || 0) + profile.timeDeltas[i] / 1000);
  });
  const cpuSamples = [...totals].sort((a, b) => b[1] - a[1]).slice(0, 30).map(([frame, selfMs]) => ({ frame, selfMs }));
  assert.deepEqual(errors, []);
  writeFileSync('reports/prototype-e-interactions.json', JSON.stringify({ measuredAt: new Date().toISOString(), baseCommit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), browser: browser.version(), renderer: 'Headless Chromium / SwiftShader; Vite development mode', viewport: { width: 1440, height: 900 }, samples, cpuSamples, errors }, null, 2) + '\n');
  console.log('Saved reports/prototype-e-interactions.json; biography verified. No screenshots.');
} finally { await browser.close(); }
