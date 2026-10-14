import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const browser = await playwright.chromium.launch({ executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--enable-unsafe-swiftshader'] });
const base = process.env.SITE_URL || 'http://localhost:5173';
const variant = process.env.VARIANT || 'b';
if (!['b', 'e'].includes(variant)) throw new Error('VARIANT must be b or e');
const duration = Number(process.env.SAMPLE_MS || 2500);
const repetitions = Number(process.env.REPETITIONS || 3);
const cases = process.env.CASES?.split(',') || ['idle', 'motion', 'nature-0', 'nature-1', 'nature-2', 'nature-3', 'signal', 'autoregressive', 'manifold', 'landscape', 'conversation', 'all', 'folded-about'];
const samples = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = []; page.on('pageerror', error => { errors.push(error.stack); console.error(error.stack); });
  const session = await page.context().newCDPSession(page);
  await session.send('Performance.enable');
  for (let repetition = 0; repetition < repetitions; repetition++) {
    // Reverse alternate runs to reduce ordering/warm-cache bias.
    for (const name of repetition % 2 ? [...cases].reverse() : cases) {
      await page.goto(`${base}/prototype/${variant}?benchmark=${name === 'folded-about' ? 'all#about' : name}`);
      await page.waitForSelector('.pb-fragment');
      const count = await page.locator('.pb-fragment').count();
      if (name !== 'all') await page.waitForFunction(expected => document.querySelectorAll('.pb-frozen').length === expected, ['idle', 'motion', 'folded-about'].includes(name) ? count : count - 1);
      await page.waitForTimeout(900);
      const before = Object.fromEntries((await session.send('Performance.getMetrics')).metrics.map(metric => [metric.name, metric.value]));
      const timing = await page.evaluate(duration => new Promise(resolve => {
        const deltas = []; let start, previous;
        const tick = now => {
          start ??= now;
          if (previous !== undefined) deltas.push(now - previous);
          previous = now;
          if (now - start < duration) return requestAnimationFrame(tick);
          deltas.sort((a,b) => a-b);
          resolve({ seconds: (now-start)/1000, fps: deltas.length*1000/(now-start), p95ms: deltas[Math.floor(deltas.length*.95)], slowFrames: deltas.filter(delta=>delta>33.5).length });
        };
        requestAnimationFrame(tick);
      }), duration);
      const after = Object.fromEntries((await session.send('Performance.getMetrics')).metrics.map(metric => [metric.name, metric.value]));
      const sample = { name, repetition, ...timing,
        mainThreadMsPerSecond: (after.TaskDuration-before.TaskDuration)*1000/timing.seconds,
        scriptMsPerSecond: (after.ScriptDuration-before.ScriptDuration)*1000/timing.seconds,
        layoutMsPerSecond: (after.LayoutDuration-before.LayoutDuration)*1000/timing.seconds,
        styleMsPerSecond: (after.RecalcStyleDuration-before.RecalcStyleDuration)*1000/timing.seconds,
        jsHeapMiB: after.JSHeapUsedSize/1024/1024,
        ...await page.evaluate(()=>({ liveCanvases: document.querySelectorAll('.pb-fragment canvas').length, frozenPreviews: document.querySelectorAll('.pb-frozen').length, runningOrbits: [...document.querySelectorAll('.pb-orbit')].filter(node=>node.getAnimations().some(animation=>animation.playState==='running')).length })),
      };
      samples.push(sample);
      console.log(`${repetition+1}/${repetitions} ${name}: ${sample.fps.toFixed(1)} FPS, ${sample.mainThreadMsPerSecond.toFixed(1)} main-thread ms/s`);
    }
  }
  if (errors.length) throw new Error(errors.join('\n'));
  const median = values => values.sort((a,b)=>a-b)[Math.floor(values.length/2)];
  const summary = cases.map(name=>{
    const rows=samples.filter(sample=>sample.name===name);
    return {name,...Object.fromEntries(['fps','p95ms','mainThreadMsPerSecond','scriptMsPerSecond','layoutMsPerSecond','styleMsPerSecond','liveCanvases','frozenPreviews','runningOrbits'].map(key=>[key,Number(median(rows.map(row=>row[key])).toFixed(2))]))};
  });
  const assets = JSON.parse(readFileSync('public/assets/images/previews/clips.json')).map(clip=>({ ...clip, bytes: statSync(`public${clip.src}`).size, decodedPixelBytes: clip.columns*Math.ceil(clip.frames/clip.columns)*clip.size*clip.size*4 }));
  const report = { measuredAt: new Date().toISOString(), baseCommit: execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(), browser: browser.version(), renderer: 'Chromium headless / SwiftShader', viewport: {width:1440,height:900}, durationMs:duration, repetitions, summary, assets, samples };
  mkdirSync('reports', {recursive:true});
  writeFileSync(`reports/prototype-${variant}-costs.json`,JSON.stringify(report,null,2)+'\n');
  const columns=Object.keys(summary[0]);
  writeFileSync(`reports/prototype-${variant}-costs.csv`,columns.join(',')+'\n'+summary.map(row=>columns.map(key=>row[key]).join(',')).join('\n')+'\n');
  console.log(`Saved reports/prototype-${variant}-costs.json and .csv`);
} finally { await browser.close(); }
