// No screenshots: compare frozen and native DOM/SVG at identical timestamps.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const { default: playwright } = await import(process.env.PLAYWRIGHT_MODULE || '/tmp/landing-browser-check/node_modules/playwright/index.js');
const base = new URL('../design/references/robust-steering-animation/frozen-9c6a434/', import.meta.url);
const read = name => readFileSync(new URL(name, base), 'utf8');
const fixture = { html: read('source-page.html'), css: read('steering-demo.css'), script: read('steering-demo.js'), timeline: read('steering-timeline.js'), examples: read('mgsm-examples.json'), fields: read('steering-landscapes.json') };
const browser = await playwright.chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, reducedMotion: 'reduce' });
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto(process.env.SITE_URL || 'http://localhost:5173/prototype/a');
  await page.evaluate(async fixture => {
    const { default: React } = await import('/node_modules/.vite/deps/react.js');
    const { default: { createRoot } } = await import('/node_modules/.vite/deps/react-dom_client.js');
    const { default: Animation } = await import('/src/shared/RobustSteeringAnimation.jsx');
    const host = document.createElement('div'); host.id = 'native-test'; host.style.cssText = 'position:fixed;left:0;top:0;width:776px;height:420px;background:white;z-index:99999'; document.body.append(host);
    const root = createRoot(host);
    window.stopSteeringTest = () => { root.unmount(); host.remove(); frame.remove(); };
    window.renderSteering = async (elapsedMs, part = 'full', compact = false, width = 776) => {
      host.style.width = width + 'px';
      root.render(React.createElement(Animation, { elapsedMs, part, compact }));
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    };
    await window.renderSteering(0);
    const markup = new DOMParser().parseFromString(fixture.html, 'text/html').querySelector('.steering-demo').outerHTML;
    const frame = document.createElement('iframe'); frame.id = 'frozen-test'; frame.style.cssText = 'position:fixed;left:800px;top:0;width:1000px;height:400px;border:0;z-index:99999';
    const script = fixture.script.replace('box.steeringDemo = { replay };', 'box.steeringDemo = { replay, seek(time) { elapsed = time; render(); } };');
    frame.srcdoc = `<style>*{box-sizing:border-box}html,body{margin:0;width:1000px;height:400px;overflow:hidden}body{--muted:#666;--teal:#007c7c;background:transparent}${fixture.css}.steering-demo{width:776px;margin:0;height:380px;background:transparent}.steering-demo-layout{height:380px;padding-bottom:20px}.steering-demo-controls{display:none}.steering-demo-state,.steering-demo-scene{opacity:1}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}</style>${markup}<script id="steering-examples" type="application/json">${JSON.stringify(JSON.parse(fixture.examples).examples)}</script><script id="steering-landscapes" type="application/json">${fixture.fields}</script><script>${fixture.timeline}</script><script>${script}</script>`;
    document.body.append(frame);
  }, fixture);
  const frame = await (await page.locator('#frozen-test').elementHandle()).contentFrame();
  await frame.waitForFunction(() => document.querySelector('.steering-demo')?.steeringDemo);
  const inspect = root => {
    const stage = root.querySelector('.steering-demo'), bounds = stage.getBoundingClientRect();
    const scale = bounds.width / 776;
    const names = ['state','language','state-map','scene','question-text','answer-text','asker','robot','stimulation','bolt','direction','direction-line','direction-head','direction-labels','route','route-line','shift-mask','base-mask'];
    return Object.fromEntries(names.map(name => {
      const node = stage.querySelector('.steering-demo-' + name), r = node.getBoundingClientRect(), style = getComputedStyle(node);
      return [name, { rect: [(r.x-bounds.x)/scale,(r.y-bounds.y)/scale,r.width/scale,r.height/scale], text: ['question-text','answer-text','language'].includes(name) ? node.textContent : undefined,
        style: Object.fromEntries(['fontFamily','fontSize','fontWeight','lineHeight','color','opacity','stroke','strokeWidth','strokeDashoffset','clipPath'].map(key=>[key,key === 'strokeDashoffset' ? Number.parseFloat(style[key].replace('calc(','')) : key === 'clipPath' && style[key].startsWith('url(') ? 'local-clip' : style[key]])),
        paths: ['state-map'].includes(name) ? [...node.querySelectorAll('.steering-demo-field-english path,.steering-demo-field-chinese path')].map(p=>p.getAttribute('d')) : undefined,
        clips: name === 'language' ? [...node.querySelectorAll('span')].map(n=>getComputedStyle(n).clipPath) : undefined,
        icons: ['asker','robot','bolt'].includes(name) ? [...node.querySelectorAll('path,circle,ellipse,rect')].map(n=>[n.tagName,...['d','cx','cy','r','rx','ry','x','y','width','height','fill','stroke','stroke-width'].map(key=>n.getAttribute(key))]) : undefined,
        glitches: name === 'answer-text' ? [...node.children].map(n=>[n.textContent,n.classList.contains('is-glitch')]) : undefined }];
    }));
  };
  const times = [0,150,500,1200,1600,2150,3000,3800,6500,6650,7300,8150,8600,8700,9400,10100,11500,13700,15900,17300,20900,23100,24500,28100,30300,31700,35300,37500,38900,42500,44700];
  for (const time of times) {
    await page.evaluate(time => window.renderSteering(time), time);
    await frame.evaluate(time => document.querySelector('.steering-demo').steeringDemo.seek(time), time);
    const native = await page.locator('#native-test').evaluate(inspect);
    const reference = await frame.locator('body').evaluate(inspect);
    for (const name of Object.keys(native)) {
      assert.equal(native[name].text, reference[name].text, `${time} ${name} text`);
      assert.deepEqual(native[name].paths, reference[name].paths, `${time} paths`);
      assert.deepEqual(native[name].glitches, reference[name].glitches, `${time} glitches`);
      assert.deepEqual(native[name].clips, reference[name].clips, `${time} heading clips`);
      assert.deepEqual(native[name].icons, reference[name].icons, `${time} icon geometry`);
      for (const key of Object.keys(native[name].style)) assert.equal(native[name].style[key], reference[name].style[key], `${time} ${name} ${key}`);
      if (!name.startsWith('route') || native.route.style.opacity !== '0') native[name].rect.forEach((value,i) => assert(Math.abs(value-reference[name].rect[i]) < .06, `${time} ${name} rect ${i}: ${value} vs ${reference[name].rect[i]}`));
    }
  }
  for (const [part, width] of [['landscape',245],['conversation',260],['full',388]]) {
    await page.evaluate(({part,width})=>window.renderSteering(9400,part,true,width),{part,width});
    await page.waitForTimeout(30);
    const root = page.locator('#native-test');
    assert.equal(await root.locator('iframe').count(),0);
    assert.equal(await root.locator('.steering-demo-state').count(),part==='conversation'?0:1);
    assert.equal(await root.locator('.steering-demo-scene').count(),part==='landscape'?0:1);
    assert.equal(await root.locator('.rs-controls').count(),0);
  }
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.evaluate(() => window.renderSteering(9400));
  await frame.evaluate(() => {
    const box = document.querySelector('.steering-demo');
    if (box.dataset.paused !== 'true') box.querySelector('.steering-demo-pause').click();
    box.steeringDemo.seek(9400);
  });
  for (const time of [0,35,70,105,140]) {
    const seekShake = (node,time) => {
      const animation = node.getAnimations()[0];
      animation.pause(); animation.currentTime = time;
      const style = getComputedStyle(node);
      return [style.transform,style.animationDuration,style.animationTimingFunction];
    };
    const native = await page.locator('#native-test .steering-demo-bolt').evaluate(seekShake,time);
    const reference = await frame.locator('.steering-demo-bolt').evaluate(seekShake,time);
    assert.deepEqual(native,reference,`lightning keyframe ${time}`);
  }
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.evaluate(() => window.stopSteeringTest());
  assert.equal(await page.locator('.rs-animation iframe').count(),0);
  await page.getByRole('button',{name:'Open Robust steering: conversation',exact:true}).click();
  const full = page.locator('dialog .rs-animation');
  assert.equal(await full.locator('.rs-stage').getAttribute('data-paused'),'true');
  assert.equal(await full.locator('.steering-demo-question-text').textContent(),'J');
  const before = await full.locator('.steering-demo-question-text').boundingBox();
  await page.clock.install();
  await full.getByRole('button',{name:'Play robust steering animation',exact:true}).click();
  await page.clock.runFor(9500);
  assert.equal(await full.locator('.rs-stage').getAttribute('data-phase'),'steering');
  const after = await full.locator('.steering-demo-question-text').boundingBox();
  assert.deepEqual(after,before);
  await full.getByRole('button',{name:'Pause robust steering animation',exact:true}).click();
  const pausedText = await full.locator('.steering-demo-answer-text').textContent();
  await page.clock.runFor(1000);
  assert.equal(await full.locator('.steering-demo-answer-text').textContent(),pausedText);
  await full.getByRole('button',{name:'Replay robust steering animation',exact:true}).click();
  assert.equal(await full.locator('.steering-demo-question-text').textContent(),'J');
  assert.equal(await full.locator('.rs-stage').getAttribute('data-paused'),'true');
  assert.equal(await page.locator('.pa-model .steering-demo-scene').count(),0);
  assert.equal(await page.locator('.pa-steering-conversation .steering-demo-state').count(),0);
  assert.deepEqual(errors,[]);
  console.log(`Passed: frozen/native text, glitch characters, SVG paths, geometry and computed styles at ${times.length} timestamps; independent native fragments at multiple sizes; no screenshots.`);
} finally { await browser.close(); }
