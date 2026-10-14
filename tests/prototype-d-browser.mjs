import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';

const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const placements = nodes => nodes.map(node => [node.dataset.piece, node.style.width, node.style.transform, node.style.getPropertyValue('--pb-angle')]);
  for (const [width, height] of [[1440,900], [1024,768], [1920,1080]]) {
    await page.setViewportSize({ width, height });
    await page.goto('http://localhost:5173/prototype/c');
    await page.waitForSelector('.pc-scene-portrait');
    await page.waitForTimeout(800);
    const frozen = await page.locator('.pb-fragment').evaluateAll(placements);
    assert.equal(await page.locator('[data-menu="research"]').textContent(), 'Research');
    assert.equal(await page.locator('[data-menu="packages"]').textContent(), 'Tool Packages');
    assert.equal(await page.locator('.pylba-animation').count(), 0);

    await page.goto('http://localhost:5173/prototype/d');
    await page.waitForSelector('.pylba-animation svg');
    await page.waitForTimeout(800);
    assert.equal(await page.locator('.pb-fragment').count(), 15);
    const adjusted = frozen.map(row => row[0] === 'manifold' ? [row[0], row[1], row[2].replace('325px', '400px'), row[3]] : row);
    assert.deepEqual(await page.locator('.pb-fragment:not([data-kind="tool"])').evaluateAll(placements), adjusted);
    assert.equal(await page.locator('[data-menu="research"]').textContent(), 'Research Publications');
    assert.equal(await page.locator('[data-menu="packages"]').textContent(), 'Tools & Packages');
    assert.equal(await page.locator('[data-piece="neuraljax-rnn"]').evaluate(n => n.style.width), '300px');
    assert.equal(await page.locator('[data-piece="neuraljax-autoencoder"]').evaluate(n => n.style.width), '234px');
    assert(await page.evaluate(() => {
      const canvas = document.querySelector('.pb-canvas');
      const name = canvas.querySelector('.pc-name-button');
      const lettering = canvas.querySelector('.pb-lettering');
      const manifold = canvas.querySelector('[data-piece="manifold"]');
      const spans = [...name.querySelectorAll('span')];
      const nameReachable = spans.every(span => {
        const r = span.getBoundingClientRect();
        return document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2)?.closest('.pc-name-button') === name;
      });
      const handCursors = [...canvas.querySelectorAll('.pb-orbit, .pb-orbit *')].every(n => getComputedStyle(n).cursor === 'pointer');
      const copyright = document.querySelector('.pb-footer > span').getBoundingClientRect();
      const controls = document.querySelector('.pb-footer .pb-scene-controls').getBoundingClientRect();
      return Number(getComputedStyle(manifold).zIndex) < Number(getComputedStyle(lettering).zIndex)
        && nameReachable && handCursors && copyright.right < controls.left;
    }), 'Name remains reachable above manifold, every preview has a hand cursor, copyright precedes right-aligned controls');
    const tool = page.locator('[data-piece="pylba"]');
    assert.equal(await tool.evaluate(node => getComputedStyle(node).opacity), '0.75');
    assert(await page.locator('.pb-fragment:not([data-kind="tool"]),.pb-lettering,.pc-scene-portrait').evaluateAll(nodes=>nodes.every(node=>getComputedStyle(node).opacity==='1')));
    assert.equal(await tool.locator('iframe,img,video,canvas,button').count(), 0);
    assert(await tool.locator('svg').evaluate(svg=>svg.querySelectorAll('animate').length>0 && !svg.animationsPaused()));
    const bounds = await tool.locator('.pb-orbit').boundingBox();
    assert(bounds.x >= 0 && bounds.y >= 0 && bounds.x+bounds.width<=width && bounds.y+bounds.height<=height);
    await tool.locator('.pb-orbit').hover();
    await page.waitForTimeout(550);
    assert.equal(await tool.evaluate(node => getComputedStyle(node).opacity), '1');
    await page.mouse.move(20,20);
    await page.waitForTimeout(550);
    assert.equal(await tool.evaluate(node => getComputedStyle(node).opacity), '0.75');
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth), false);
  }
  const source = readFileSync(new URL('../src/shared/pylba/pylba.svg', import.meta.url), 'utf8');
  assert(await page.locator('.pylba-animation svg').evaluate((svg, source)=>{
    const original = new DOMParser().parseFromString(source,'image/svg+xml');
    const paths = node=>Array.from(node.querySelectorAll('path'),p=>p.getAttribute('d'));
    const animations = node=>Array.from(node.querySelectorAll('animate'),n=>Array.from(n.attributes,a=>[a.name,a.value]));
    return JSON.stringify(paths(svg))===JSON.stringify(paths(original)) && JSON.stringify(animations(svg))===JSON.stringify(animations(original));
  },source),'Original path geometry and SMIL samples/timing are preserved');
  const svg = page.locator('.pylba-animation svg');
  const motion = await svg.evaluate(node=>{
    node.pauseAnimations(); node.setCurrentTime(.05);
    const first=Array.from(node.querySelectorAll('.onion path'),p=>getComputedStyle(p).opacity);
    node.setCurrentTime(.25);
    const second=Array.from(node.querySelectorAll('.onion path'),p=>getComputedStyle(p).opacity);
    node.unpauseAnimations();
    return JSON.stringify(first)!==JSON.stringify(second);
  });
  assert(motion, 'SVG samples visibly change over time');
  await page.locator('[data-menu="packages"]').hover();
  assert.equal(await page.locator('#pb-packages-menu a[href="#pylba"] small').textContent(), '2025');
  assert(await page.evaluate(() => {
    const research = document.querySelector('#pb-research-menu');
    const packages = document.querySelector('#pb-packages-menu');
    return ['', ' a', ' small'].every(selector => {
      const first = getComputedStyle(selector ? research.querySelector(selector.trim()) : research);
      const second = getComputedStyle(selector ? packages.querySelector(selector.trim()) : packages);
      return ['fontFamily', 'fontSize', 'lineHeight', 'padding', 'color'].every(property => first[property] === second[property]);
    }) && getComputedStyle(research).width === getComputedStyle(packages).width;
  }), 'Tool dropdown uses the same typography, spacing and width as research');
  await page.locator('#pb-packages-menu a[href="#pylba"]').click();
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pb-tool h2').textContent(), '\u{1f680} pyLBA: Linear Ballistic Accumulator Models in Python');
  assert.deepEqual(await page.locator('.pb-tool .pb-project-links a').allTextContents(), [' code', 'Documentation']);
  assert.equal(await page.locator('.pb-tool .pb-project-links img').evaluate(n => getComputedStyle(n).width), '20px');
  assert.equal(await page.locator('.pb-tool .pb-authors').textContent(), 'Rudramani Singha');
  assert.equal(await page.locator('.pb-tool .pb-blurb').textContent(), readFileSync(new URL('../src/content/pylba.md', import.meta.url),'utf8').trim());
  assert(await page.locator('.pb-tool').evaluate(n=>n.children[1].classList.contains('pb-authors') && n.children[2].classList.contains('pb-project-links') && n.children[3].classList.contains('pb-tool-body')));
  for (const [width, height] of [[1024,768], [1440,900], [1920,1080]]) {
    await page.setViewportSize({ width, height });
    const layout = await page.locator('.pb-tool-body').evaluate(body => {
      const media = body.querySelector('.pb-tool-media');
      const figure = media.getBoundingClientRect();
      const paragraph = body.querySelector('.pb-blurb p');
      const range = document.createRange();
      range.selectNodeContents(paragraph);
      const lines = Array.from(range.getClientRects());
      return {
        floated: getComputedStyle(media).cssFloat === 'left',
        aligned: Math.abs(figure.left - body.getBoundingClientRect().left) < 1,
        beside: lines[0].left >= figure.right + 27,
        originalSize: Math.abs(figure.width - Math.min(300, body.clientWidth * .4)) < 1,
        clear: lines.every(line => line.top >= figure.bottom || line.left >= figure.right + 27),
        contained: body.getBoundingClientRect().bottom >= figure.bottom,
        overflow: document.documentElement.scrollWidth > innerWidth,
      };
    });
    assert.deepEqual(layout, { floated: true, aligned: true, beside: true, originalSize: true, clear: true, contained: true, overflow: false }, `Wrapped figure at ${width}px`);
  }
  assert.equal(await page.locator('.pb-fragment animate').count(), 0, 'Folded previews are inert');
  const full = page.locator('.pb-tool .pylba-animation');
  await full.getByRole('button',{name:'Pause PyLBA animation'}).click();
  assert(await full.locator('svg[role="img"]').evaluate(n=>n.animationsPaused()));
  await full.locator('svg[role="img"]').evaluate(n=>n.setCurrentTime(.6));
  await full.getByRole('button',{name:'Replay PyLBA animation'}).click();
  assert(await full.locator('svg[role="img"]').evaluate(n=>n.getCurrentTime()<.02 && n.animationsPaused()));
  await full.getByRole('button',{name:'Play PyLBA animation',exact:true}).click();
  assert(await full.locator('svg[role="img"]').evaluate(n=>!n.animationsPaused()));
  await page.keyboard.press('Escape');
  await page.waitForTimeout(850);
  await page.locator('[data-piece="pylba"] .pb-orbit').click();
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pb-tool').count(), 1);
  await page.reload();
  await page.waitForSelector('.pb-tool .pylba-animation svg');
  assert.equal(await page.locator('.pb-tool').count(), 1, 'Direct tool hash opens its page');
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.waitForFunction(()=>document.querySelector('.pb-tool .pylba-animation > svg')?.animationsPaused());
  assert(await page.locator('.pb-tool .pylba-animation > svg').evaluate(n=>n.animationsPaused()));
  assert.deepEqual(errors, []);
  console.log('Passed: preserved C, additive D layout at three sizes, opacity, source SVG parity, live motion, tool navigation, pause/replay, reduced motion and inert stack. No screenshots.');
} finally { await browser.close(); }
