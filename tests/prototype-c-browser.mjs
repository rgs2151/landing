import assert from 'node:assert/strict';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';
import { createLocalMotion } from '../src/prototypes/c/localMotion.js';

for (let index=0; index<9; index++) {
  const motion=createLocalMotion(index);
  assert.deepEqual(motion,createLocalMotion(index));
  assert.equal(motion.keyframes[0].transform,motion.keyframes.at(-1).transform);
  for (const frame of motion.keyframes) {
    const [x,y]=frame.transform.match(/-?\d+(?:\.\d+)?/g).map(Number);
    assert(Math.abs(x)<=12 && Math.abs(y)<=9);
    assert(!frame.transform.includes('scale') && !('opacity' in frame));
  }
}
const browser=await playwright.chromium.launch({executablePath:'/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',args:['--enable-unsafe-swiftshader']});
try {
  const page=await browser.newPage();
  const errors=[]; page.on('pageerror',error=>errors.push(error.message));
  for (const [width,height] of [[1440,900],[1024,768],[1920,1080]]) {
    await page.setViewportSize({width,height});
    await page.goto('http://localhost:5173/prototype/c');
    await page.waitForSelector('.prototype-c');
    await page.waitForFunction(()=>document.querySelectorAll('.pb-nature-preview canvas[data-ready="true"]').length===4);
    await page.waitForTimeout(850);
    assert.deepEqual(await page.locator('.pb-fragment').evaluateAll(nodes=>nodes.map(n=>parseFloat(n.style.width))),[216,125,95,65,270,350,150,310,380]);
    assert.deepEqual(await page.locator('[data-piece="nature-0"],[data-piece="nature-1"],[data-piece="signal"],[data-piece="autoregressive"]').evaluateAll(nodes=>nodes.map(node=>{
      const m=new DOMMatrixReadOnly(node.style.transform);return [node.dataset.piece,m.m41,m.m42];
    })),[['nature-0',755,105],['nature-1',205,-15],['signal',785,525],['autoregressive',505,100]]);
    assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'home');
    assert.equal(await page.locator('.pc-scene-portrait img').count(),1);
    assert(await page.evaluate(()=>{
      const title=document.querySelector('.pb-title'),span=title.querySelector('span'),range=document.createRange();
      range.setStart(span.firstChild,0);range.setEnd(span.firstChild,1);
      const r=range.getBoundingClientRect();
      return title.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));
    }),'The opening R stays unobstructed');
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    const hidden = await page.locator('.pb-orbit').evaluateAll(nodes=>nodes.flatMap(node=>{
      const animation=node.getAnimations()[0]; animation.pause();
      const duration=animation.effect.getTiming().duration;
      return [0,.25,.5,.75].flatMap(phase=>{
        animation.currentTime=duration*phase;
        const box=node.getBoundingClientRect();
        const hit=document.elementFromPoint(box.x+box.width/2,box.y+box.height/2);
        return node.closest('.pb-fragment').contains(hit) ? [] : [{id:node.closest('.pb-fragment').dataset.piece,phase,hit:hit?.className}];
      });
    }));
    assert.deepEqual(hidden,[],`${width}: every preview's center stays accessible through the local movement`);
    await page.locator('[data-piece="manifold"] .pb-orbit').hover();
    await page.waitForTimeout(550);
    assert.equal(await page.locator('.pb-canvas').getAttribute('data-active'),'neural-field-manifold');
    assert(await page.locator('.pb-orbit').evaluateAll(nodes=>nodes.every(n=>n.getAnimations()[0].playState==='paused')));
    await page.mouse.move(20,20);
  }
  for (const id of ['cooperation','robust-steering','neural-field-manifold']) {
    await page.locator('[data-menu="research"]').hover();
    await page.locator(`.pb-navigation a[href="#${id}"]`).click();
    await page.waitForTimeout(850);
    assert.equal(await page.locator('.pb-project').count(),1);
    assert.equal(await page.locator('.pb-fragment canvas').count(),0);
    assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'stack');
    assert(await page.evaluate(async id=>{
      const content=await import('/src/content/site.js');
      const project=id==='cooperation'?content.paper:id==='robust-steering'?content.robustSteering:content.neuralFieldManifold;
      return document.querySelector('.pb-blurb').textContent.replace(/\s/g,'')===project.blurb.replace(/\s/g,'');
    },id));
    await page.keyboard.press('Escape');
    await page.waitForTimeout(850);
  }
  await page.locator('.pb-navigation > a[href="#about"]').click();
  await page.waitForTimeout(850);
  assert(await page.locator('.pb-fragment').evaluateAll(nodes=>nodes.every(n=>getComputedStyle(n).visibility==='hidden')));
  assert.equal(await page.locator('.pc-scene-portrait img').evaluate(n=>getComputedStyle(n).filter),'none');
  assert.equal(await page.locator('.pc-scene-portrait button').count(),0);
  const aboutPhoto=await page.locator('.pc-scene-portrait').boundingBox();
  await page.locator('.pc-scene-portrait img').evaluate(n=>n.dataset.continuity='same-photo');
  await page.getByRole('button',{name:'Return to composition',exact:true}).click();
  const returnSource=await page.locator('.pc-scene-portrait').evaluate(node=>{
    const matrix=new DOMMatrixReadOnly(node.getAnimations()[0].effect.getKeyframes()[0].transform);
    return {x:parseFloat(node.style.left)+matrix.m41,y:parseFloat(node.style.top)+matrix.m42};
  });
  assert(Math.abs(returnSource.x-aboutPhoto.x)<2 && Math.abs(returnSource.y-aboutPhoto.y)<2,'Return starts from About, not the bottom-left stack');
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'home');
  assert.equal(await page.locator('.pc-scene-portrait img').getAttribute('data-continuity'),'same-photo');
  assert.equal(await page.locator('.pc-scene-portrait').evaluate(n=>getComputedStyle(n).visibility),'visible');
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.locator('.pb-navigation > a[href="#about"]').click();
  await page.keyboard.press('Escape');
  await page.waitForTimeout(100);
  assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'home');
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.goto('http://localhost:5173/prototype/b');
  await page.waitForSelector('.pb-fragment');
  assert.equal(await page.locator('.prototype-c').count(),0);
  assert.equal(await page.locator('[data-piece="manifold"]').evaluate(n=>parseFloat(n.style.width)),456);
  assert.equal(await page.locator('.pb-orbit').first().evaluate(n=>n.getAnimations()[0].effect.getKeyframes().length),181,'B keeps its existing depth-drift motion');
  assert.deepEqual(errors,[]);
  console.log('Passed: C spread and visibility at three desktop sizes, bounded local movement, hover, frozen project content, fold/back/About, and unchanged B defaults. No screenshots.');
} finally {await browser.close();}
