import assert from 'node:assert/strict';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';
import { createLocalMotion } from '../src/prototypes/c/localMotion.js';

for (let index=0; index<9; index++) {
  const motion=createLocalMotion(index);
  assert.deepEqual(motion,createLocalMotion(index));
  assert.equal(motion.duration,18000+index*900);
  assert.equal(motion.keyframes[0].transform,motion.keyframes.at(-1).transform);
  for (const frame of motion.keyframes) {
    const [x,y]=frame.transform.match(/-?\d+(?:\.\d+)?/g).map(Number);
    assert(Math.abs(x)<=10 && Math.abs(y)<=7.5);
    assert(!frame.transform.includes('scale') && !('opacity' in frame));
  }
}
const browser=await playwright.chromium.launch({executablePath:'/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',args:['--enable-unsafe-swiftshader']});
async function hasManifoldPixels(canvas) {
  return canvas.evaluate(node=>new Promise(resolve=>{
    const gl=node.getContext('webgl2') || node.getContext('webgl');
    let attempts=0;
    const sample=()=>{
      const pixels=new Uint8Array(node.width*node.height*4);
      gl.readPixels(0,0,node.width,node.height,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
      if(pixels.some((value,index)=>index%4===3 && value>0)) resolve(true);
      else if(++attempts<60) requestAnimationFrame(sample);
      else resolve(false);
    };
    requestAnimationFrame(sample);
  }));
}
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
    })),[['nature-0',690,65],['nature-1',170,120],['signal',780,515],['autoregressive',365,80]]);
    assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'home');
    assert.equal(await page.locator('.pc-scene-portrait img').count(),1);
    assert.equal(await page.locator('[data-piece="manifold"] .nf-manifold').getAttribute('data-stroke-width'),'2');
    assert(await hasManifoldPixels(page.locator('[data-piece="manifold"] .nf-manifold canvas')),'Shared preview manifold renders');
    assert(await page.evaluate(()=>{
      const z=selector=>Number(getComputedStyle(document.querySelector(selector)).zIndex);
      return z('[data-piece="nature-0"]')<z('.pc-scene-portrait') && z('[data-piece="landscape"]')>z('.pc-scene-portrait');
    }),'Large footage sits behind the portrait; steering landscape stays above it');
    assert(await page.evaluate(()=>{
      const portrait=document.querySelector('.pc-scene-portrait'), image=portrait.querySelector('img');
      const box=portrait.getBoundingClientRect(), title=document.querySelector('.pb-title').getBoundingClientRect();
      const right=Math.max(...Array.from(document.querySelectorAll('.pb-title span'),n=>n.getBoundingClientRect().right));
      return Math.abs(box.width-box.height)<1 && Math.abs(box.height-title.height)<1
        && Math.abs(box.y-title.y)<1 && box.x>right
        && image.naturalWidth===image.naturalHeight && getComputedStyle(image).objectFit==='contain';
    }),'Full square photo matches the two-line name height and sits beside it');
    assert(await page.evaluate(()=>{
      const title=document.querySelector('.pb-title');
      return Array.from(title.querySelectorAll('span')).every(span=>Array.from(span.textContent).every((_,index)=>{
        const range=document.createRange();range.setStart(span.firstChild,index);range.setEnd(span.firstChild,index+1);
        const r=range.getBoundingClientRect();
        return title.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));
      }));
    }),'Name lettering stays readable through the overlaps');
    assert(await page.evaluate(()=>{
      const portrait=document.querySelector('.pc-scene-portrait'),box=portrait.getBoundingClientRect();
      return [[.36,.22],[.64,.22],[.36,.57],[.64,.57],[.5,.4]].every(([x,y])=>
        portrait.contains(document.elementFromPoint(box.x+box.width*x,box.y+box.height*y)));
    }),'Portrait face stays unobstructed');
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
    for(const selector of ['.pc-name-button','.pc-scene-portrait button']) {
      const box=await page.locator(selector).boundingBox();
      await page.mouse.move(box.x+box.width/2,box.y+box.height/2);
      await page.waitForTimeout(400);
      assert.equal(await page.locator('.pb-canvas').getAttribute('data-active'),'about');
      assert(await page.locator('.pb-fragment').evaluateAll(nodes=>nodes.every(n=>Number(getComputedStyle(n).opacity)===.25)));
      assert(await page.locator('.pb-lettering,.pc-scene-portrait').evaluateAll(nodes=>nodes.every(n=>getComputedStyle(n).opacity==='1')));
      assert(await page.locator('.pb-orbit').evaluateAll(nodes=>nodes.every(n=>n.getAnimations()[0].playState==='paused')));
      await page.mouse.move(20,20);
      await page.waitForTimeout(400);
      assert.equal(await page.locator('.pb-canvas').getAttribute('data-active'),'');
      assert(await page.locator('.pb-fragment').evaluateAll(nodes=>nodes.every(n=>getComputedStyle(n).opacity==='1')));
    }
  }
  for (const id of ['cooperation','robust-steering','neural-field-manifold']) {
    await page.locator('[data-menu="research"]').hover();
    await page.locator(`.pb-navigation a[href="#${id}"]`).click();
    await page.waitForTimeout(850);
    assert.equal(await page.locator('.pb-project').count(),1);
    assert.equal(await page.locator('.pb-fragment canvas').count(),0);
    assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'stack');
    if(id==='neural-field-manifold') {
      assert.equal(await page.locator('.pb-project .nf-manifold').getAttribute('data-stroke-width'),'2');
      assert(await hasManifoldPixels(page.locator('.pb-project .nf-manifold canvas')),'Shared full manifold renders');
    }
    assert(await page.evaluate(async id=>{
      const content=await import('/src/content/site.js');
      const project=id==='cooperation'?content.paper:id==='robust-steering'?content.robustSteering:content.neuralFieldManifold;
      return document.querySelector('.pb-blurb').textContent.replace(/\s/g,'')===project.blurb.replace(/\s/g,'');
    },id));
    await page.keyboard.press('Escape');
    await page.waitForTimeout(850);
  }
  await page.locator('.pc-name-button').click();
  await page.waitForTimeout(850);
  assert(await page.locator('.pb-fragment').evaluateAll(nodes=>nodes.every(n=>getComputedStyle(n).visibility==='visible')));
  assert.equal(await page.locator('.pc-scene-portrait img').evaluate(n=>getComputedStyle(n).filter),'none');
  assert.equal(await page.locator('.pc-scene-portrait button').count(),0);
  const aboutPhoto=await page.locator('.pc-scene-portrait').boundingBox();
  const aboutTitle=await page.locator('.pb-about h2').boundingBox();
  assert(Math.abs(aboutPhoto.y-aboutTitle.y)<1 && aboutPhoto.x<aboutTitle.x,'About portrait starts at the heading level on the left');
  assert.equal(await page.locator('.pb-about-photo-slot').evaluate(n=>getComputedStyle(n).float),'left');
  await page.locator('.pc-scene-portrait img').evaluate(n=>n.dataset.continuity='same-photo');
  await page.getByRole('button',{name:'Return to composition',exact:true}).click();
  const returnSource=await page.locator('.pc-scene-portrait').evaluate(node=>{
    const matrix=new DOMMatrixReadOnly(node.getAnimations()[0].effect.getKeyframes()[0].transform);
    const scene=node.closest('.pb-canvas'),box=scene.getBoundingClientRect(),scale=box.width/1200;
    return {x:box.x+(parseFloat(node.style.left)+matrix.m41)*scale,y:box.y+(parseFloat(node.style.top)+matrix.m42)*scale};
  });
  assert(Math.abs(returnSource.x-aboutPhoto.x)<2 && Math.abs(returnSource.y-aboutPhoto.y)<2,'Return starts from About, not the bottom-left stack');
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'home');
  assert.equal(await page.locator('.pc-scene-portrait img').getAttribute('data-continuity'),'same-photo');
  assert.equal(await page.locator('.pc-scene-portrait').evaluate(n=>getComputedStyle(n).visibility),'visible');
  const homePhoto=await page.locator('.pc-scene-portrait button').boundingBox();
  await page.mouse.click(homePhoto.x+homePhoto.width/2,homePhoto.y+homePhoto.height/2);
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'about');
  const stackPoint=await page.locator('.pb-orbit').last().evaluate(n=>{
    const box=n.getBoundingClientRect(),x=box.x+box.width/2,y=box.y+box.height/2;
    const hit=document.elementFromPoint(x,y);
    return {x,y,cursor:getComputedStyle(hit).cursor,clickable:Boolean(hit.closest('.pb-fragment'))};
  });
  assert(stackPoint.clickable && stackPoint.cursor==='pointer','About stack has an active pointer target');
  await page.mouse.click(stackPoint.x,stackPoint.y);
  await page.waitForTimeout(850);
  assert.equal(await page.locator('.pc-scene-portrait').getAttribute('data-location'),'home');
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
