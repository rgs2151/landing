import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const browser = await playwright.chromium.launch({ executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://localhost:5173/prototype/b');
  await page.waitForFunction(() => document.querySelectorAll('.pb-nature-preview canvas[data-ready="true"]').length === 4);
  await page.waitForTimeout(800);
  assert.deepEqual(await page.locator('.prototype-b,.pb-title,.pb-title span').evaluateAll(nodes => nodes.map(node => ({ background: getComputedStyle(node).backgroundColor, color: getComputedStyle(node).color }))), [
    { background: 'rgb(255, 255, 255)', color: 'rgb(17, 17, 17)' },
    { background: 'rgba(0, 0, 0, 0)', color: 'rgb(255, 255, 255)' },
    { background: 'rgb(0, 0, 0)', color: 'rgb(255, 255, 255)' },
    { background: 'rgb(0, 0, 0)', color: 'rgb(255, 255, 255)' },
  ]);
  assert.deepEqual(await page.locator('.pb-fragment').evaluateAll(nodes=>nodes.map(node=>parseFloat(node.style.width))),[324,150,114,78,324,420,180,372,456], 'All nine assets remain, each 20% larger');
  const radii = await page.locator('.pb-orbit').evaluateAll(nodes => nodes.map(node => {
    const frames = node.getAnimations()[0].effect.getKeyframes().map(frame => new DOMMatrixReadOnly(frame.transform));
    return (Math.max(...frames.map(m=>m.m41))-Math.min(...frames.map(m=>m.m41)))/2;
  }));
  assert.equal(new Set(radii.map(radius=>Math.round(radius))).size,9,'Every asset has a distinct orbit radius');
  assert(Math.abs(radii.reduce((sum,radius)=>sum+radius,0)/9-372)<1,'Mean orbit radius is 20% larger');
  const depth = await page.locator('.pb-fragment,.pb-lettering').evaluateAll(nodes => Object.fromEntries(nodes.map(node => [node.dataset.piece || 'name', +getComputedStyle(node).zIndex])));
  for (const id of ['autoregressive','manifold','nature-0']) assert(depth[id] > depth.name);
  for (const id of ['conversation','nature-1','nature-2','nature-3']) assert(depth[id] < depth.name);
  const orbit = page.locator('[data-piece="manifold"] .pb-orbit');
  const cycle = await orbit.evaluate(node => {
    const animation = node.getAnimations()[0], duration = animation.effect.getTiming().duration;
    animation.pause();
    return [0,.25,.5,.75,1].map(fraction => {
      animation.currentTime = duration * fraction;
      const matrix = new DOMMatrixReadOnly(getComputedStyle(node).transform);
      return [matrix.m41, matrix.m42];
    });
  });
  assert(Math.hypot(cycle[0][0]-cycle[2][0],cycle[0][1]-cycle[2][1]) > 300, 'Motion completes a full orbit, not a sway');
  assert(Math.hypot(cycle[0][0]-cycle[4][0],cycle[0][1]-cycle[4][1]) < 1, 'Orbit loops seamlessly');
  await page.locator('[data-menu="research"]').hover();
  await page.locator('.pb-navigation a[href="#robust-steering"]').hover();
  await page.waitForTimeout(800);
  assert(await page.locator('.pb-orbit').evaluateAll(nodes => nodes.every(node => node.getAnimations()[0].playState === 'paused')));
  assert(await page.locator('.pb-fragment[data-matching="true"] .pb-tilt').evaluateAll(nodes => nodes.every(node => Math.abs(new DOMMatrixReadOnly(getComputedStyle(node).transform).m12) < .001)), 'Hovered project straightens');
  assert(await page.locator('.pb-fragment:not([data-matching="true"])').evaluateAll(nodes => nodes.every(node => +getComputedStyle(node).opacity === .25)));
  assert.equal(await page.locator('.pb-lettering').evaluate(node=>getComputedStyle(node).opacity),'1');
  assert(await page.locator('[data-piece="landscape"] .steering-demo-axis-labels text').evaluateAll(nodes => nodes.every(node => {
    const box = node.getBoundingClientRect(), outer = node.closest('.rs-animation').getBoundingClientRect();
    return box.left >= outer.left && box.right <= outer.right && box.top >= outer.top && box.bottom <= outer.bottom;
  })), 'Both axis labels fit within the padded preview');
  await page.locator('.pb-navigation a[href="#robust-steering"]').click();
  await page.waitForTimeout(800);
  assert.equal(await page.locator('.prototype-b').evaluate(node=>getComputedStyle(node).backgroundColor),'rgb(255, 255, 255)');
  await page.locator('.pb-traveling-portrait button').click();
  assert.equal(await page.locator('.pb-canvas').getAttribute('data-transitioning'),'true');
  await page.mouse.move(500,400);
  await page.mouse.move(850,450,{steps:12});
  assert.equal(await page.locator('.pb-canvas').getAttribute('data-active'),'');
  assert(await page.locator('.pb-orbit').evaluateAll(nodes => nodes.every(node=>getComputedStyle(node).pointerEvents==='none')));
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pb-canvas').getAttribute('data-transitioning'),'false');
  await page.locator('.pb-navigation > a[href="#about"]').click();
  await page.waitForTimeout(850);
  assert(await page.locator('.pb-fragment').evaluateAll(nodes=>nodes.every(node=>getComputedStyle(node).visibility==='hidden')));
  await page.keyboard.press('Escape');
  await page.waitForTimeout(850);
  await page.locator('[data-menu="research"]').hover();
  await page.locator('.pb-navigation a[href="#robust-steering"]').hover();
  await page.waitForTimeout(800);

  if (process.env.SKIP_BENCHMARK === '1') {
    assert.deepEqual(errors,[]);
    console.log('Passed: nine enlarged assets, distinct enlarged orbits, text-only highlights, hover, labels, transitions and About. No screenshots.');
  } else {
  // Isolate opacity cost: identical hovered scene, paused orbits, active figures.
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Performance.enable');
  const samples = [];
  for (let repetition=0; repetition<3; repetition++) {
    for (const dimmed of repetition%2 ? [true,false] : [false,true]) {
      const override = dimmed ? null : await page.addStyleTag({content:'.pb-fragment { opacity: 1 !important; }'});
      await page.waitForTimeout(400);
      const before = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m=>[m.name,m.value]));
      const timing = await page.evaluate(() => new Promise(resolve => {
        const intervals=[]; let start,previous;
        function tick(now) {
          start ??= now;
          if (previous !== undefined) intervals.push(now-previous);
          previous=now;
          if (now-start<1500) return requestAnimationFrame(tick);
          intervals.sort((a,b)=>a-b);
          resolve({seconds:(now-start)/1000,fps:intervals.length*1000/(now-start),p95ms:intervals[Math.floor(intervals.length*.95)]});
        }
        requestAnimationFrame(tick);
      }));
      const after = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map(m=>[m.name,m.value]));
      samples.push({repetition,dimmed,...timing,mainThreadMsPerSecond:(after.TaskDuration-before.TaskDuration)*1000/timing.seconds});
      if (override) await override.evaluate(node=>node.remove());
    }
  }
  assert.deepEqual(errors,[]);
  const median = values => values.sort((a,b)=>a-b)[Math.floor(values.length/2)];
  const summary = [false,true].map(dimmed=>({dimmed,...Object.fromEntries(['fps','p95ms','mainThreadMsPerSecond'].map(key=>[key,+median(samples.filter(s=>s.dimmed===dimmed).map(s=>s[key])).toFixed(2)]))}));
  writeFileSync('reports/prototype-b-focus.json',JSON.stringify({measuredAt:new Date().toISOString(),browser:browser.version(),renderer:'Headless Chromium / SwiftShader',viewport:{width:1440,height:900},sampleMs:1500,summary,samples},null,2)+'\n');
  console.log(JSON.stringify(summary,null,2));
  console.log('Passed: splash contrast, depth, full orbit, hover freeze/straightening, axis padding, transition lock and hidden About stack. No screenshots.');
  }
} finally { await browser.close(); }
