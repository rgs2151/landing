import assert from 'node:assert/strict';
const { default: playwright } = await import(process.env.PLAYWRIGHT_MODULE || '/tmp/landing-browser-check/node_modules/playwright/index.js');
const browser = await playwright.chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell', args: ['--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage();
  const errors = []; page.on('pageerror',error=>errors.push(error.message));
  const base = process.env.SITE_URL || 'http://localhost:5173';
  for (const [width,height] of [[1440,900],[1024,768],[1920,1080]]) {
    await page.setViewportSize({width,height});
    await page.goto(base+'/prototype/b');
    await page.waitForFunction(()=>document.querySelectorAll('.pb-fragment').length===9);
    await page.waitForTimeout(850);
    const audit = await page.evaluate(()=>{
      const targets=[...document.querySelectorAll('.pb-title,.pb-fragment,.pb-navigation > a,.pb-nav-group > button')];
      const boxes=targets.map(node=>{
        const r=node.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
        return {name:node.getAttribute('aria-label')||node.textContent,blocked:!node.contains(hit),overflow:!node.matches('.pb-fragment')&&node.scrollWidth>node.clientWidth+1,offscreen:r.x<0||r.right>innerWidth};
      });
      const collisions=[];
      for(const title of document.querySelectorAll('.pb-title')) {
        const walker=document.createTreeWalker(title,NodeFilter.SHOW_TEXT);
        while(walker.nextNode()) {
          if(!walker.currentNode.textContent.trim())continue;
          const range=document.createRange(); range.selectNode(walker.currentNode);
          for(const text of range.getClientRects())for(const asset of document.querySelectorAll('.pb-fragment')) {
            const r=asset.getBoundingClientRect();
            if(Math.min(text.right,r.right)-Math.max(text.left,r.left)>1 && Math.min(text.bottom,r.bottom)-Math.max(text.top,r.top)>1)collisions.push([walker.currentNode.textContent,asset.getAttribute('aria-label')]);
          }
        }
      }
      return {boxes,collisions,overflow:document.documentElement.scrollWidth>innerWidth,playing:[...document.querySelectorAll('.pb-nature-preview canvas')].map(v=>v.dataset.ready==='true'&&Number(v.dataset.frame)>0)};
    });
    assert.equal(audit.overflow,false);
    assert.deepEqual(audit.collisions,[],`${width} title/figure collisions`);
    assert.deepEqual(audit.boxes.filter(item=>item.blocked||item.overflow||item.offscreen),[],JSON.stringify({width,audit}));
    assert.deepEqual(audit.playing,[true,true,true,true]);
    assert.equal(await page.locator('.pb-fragment video').count(),0);
    const frames = () => page.locator('.pb-nature-preview canvas').evaluateAll(nodes=>nodes.map(node=>node.dataset.frame));
    const playingFrames = await frames();
    await page.waitForTimeout(200);
    assert.notDeepEqual(await frames(),playingFrames,'Cached footage advances');
    assert(await page.locator('.pb-nature-preview canvas').evaluateAll(nodes=>nodes.every(node=>{
      const pixels=node.getContext('2d').getImageData(0,0,node.width,node.height).data;
      return pixels.some((value,index)=>index%4!==3&&value<200);
    })),'All four cached clips have nonblank frames');
    assert.equal(await page.locator('dialog,[role="dialog"],iframe').count(),0);
    const before=await page.locator('.pb-fragment[data-project="robust-steering"]').first().getAttribute('style');
    assert.equal(await page.locator('.pb-identity,.pb-nature-mark,.pb-footer > button').count(),0);
    assert.equal(await page.locator('.pb-title').innerText(),'Rudramani\nSingha');
    await page.locator('[data-menu="research"]').hover();
    await page.locator('.pb-navigation a[href="#robust-steering"]').hover();
    const after=await page.locator('.pb-fragment[data-project="robust-steering"]').first().getAttribute('style');
    assert.notEqual(before,after);
    for(const id of ['cooperation','robust-steering','neural-field-manifold']) {
      await page.locator('[data-menu="research"]').hover();
      await page.locator(`.pb-navigation a[href="#${id}"]`).click();
      await page.waitForTimeout(800);
      assert.equal(new URL(page.url()).hash,'#'+id);
      assert.equal(await page.locator('dialog,[role="dialog"]').count(),0);
      const data=await page.evaluate(async id=>{
        const site=await import('/src/content/site.js');
        const content=id==='cooperation'?site.paper:id==='robust-steering'?site.robustSteering:site.neuralFieldManifold;
        const media=document.querySelector('.pb-project-media').getBoundingClientRect(),h=document.querySelector('#pb-project-heading').getBoundingClientRect();
        const article=document.querySelector('.pb-project').getBoundingClientRect();
        const authors=document.querySelector('.pb-authors').getBoundingClientRect(),links=document.querySelector('.pb-project-links').getBoundingClientRect(),blurb=document.querySelector('.pb-blurb').getBoundingClientRect();
        return {sameTitle:document.querySelector('#pb-project-heading').textContent===content.title,sameBlurb:document.querySelector('.pb-blurb').textContent.replace(/\s/g,'')===content.blurb.replace(/\s/g,''),overlap:h.bottom>authors.top||authors.bottom>links.top||links.bottom>media.top||media.bottom>blurb.top,overflow:article.right>innerWidth,backgroundPaused:[...document.querySelectorAll('.pb-fragment video')].every(v=>v.paused),focus:document.activeElement.id,fullWidth:Math.abs(blurb.width-article.width)<1,justified:getComputedStyle(document.querySelector('.pb-blurb')).textAlign==='justify',font:getComputedStyle(document.querySelector('.pb-project')).fontFamily};
      },id);
      assert.deepEqual(data,{sameTitle:true,sameBlurb:true,overlap:false,overflow:false,backgroundPaused:true,focus:'pb-project-heading',fullWidth:true,justified:true,font:'Arial, Helvetica, sans-serif'});
      const stopped = await frames();
      await page.waitForTimeout(150);
      assert.deepEqual(await frames(),stopped,'Collapsed preview playback stops');
      assert.equal(await page.locator('.pb-identity').count(),1);
      if(id==='neural-field-manifold') {
        const pixels=await page.locator('.pb-project canvas').evaluate(async canvas=>{
          let result=0;
          for(let i=0;i<6&&!result;i++) await new Promise(resolve=>requestAnimationFrame(()=>{
            const gl=canvas.getContext('webgl2')||canvas.getContext('webgl');
            if(gl){const buffer=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,buffer);for(let j=3;j<buffer.length;j+=4)if(buffer[j])result++;}
            else result=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data.some(x=>x>0)?1:0;
            resolve();
          }));
          return result;
        });
        assert(pixels>0,'Manifold canvas must contain rendered pixels');
      }
      await page.getByRole('button',{name:'Return to composition',exact:true}).click();
      await page.waitForTimeout(800);
      assert.equal(await page.locator('.pb-reading').count(),0);
    }
    await page.locator('.pb-title').click();
    await page.waitForSelector('.pb-about');
    assert.equal(await page.locator('.pb-about h2').innerText(),'Rudramani Singha');
    await page.getByRole('button',{name:'Return to composition',exact:true}).click();
    await page.waitForTimeout(750);
    await page.getByRole('button',{name:'Pause composition',exact:true}).click();
    const pausedFrames = await frames();
    await page.waitForTimeout(200);
    assert.deepEqual(await frames(),pausedFrames);
    await page.getByRole('button',{name:'Replay composition',exact:true}).click();
    await page.waitForFunction(()=>[...document.querySelectorAll('.pb-nature-preview canvas')].every(node=>node.dataset.frame==='0'));
    await page.getByRole('button',{name:'Play composition',exact:true}).click();
  }
  await page.goto(base+'/prototype/b#robust-steering');
  await page.waitForSelector('#pb-project-heading');
  await page.locator('[data-menu="research"]').hover();
  assert.match(await page.locator('.pb-navigation a[href="#neural-field-manifold"]').innerText(),/NeurIPS 2026.*Accepted/);
  assert.match(await page.locator('.pb-navigation a[href="#robust-steering"]').innerText(),/Under review/);
  assert.doesNotMatch(await page.locator('.pb-navigation').innerText(),/ICLR/);
  await page.locator('.pb-navigation a[href="#neural-field-manifold"]').click();
  await page.goBack();
  await page.waitForFunction(()=>document.querySelector('.prototype-b')?.dataset.selected==='robust-steering');
  await page.keyboard.press('Escape');
  await page.waitForFunction(()=>!document.querySelector('.pb-reading'));
  await page.emulateMedia({reducedMotion:'reduce'});
  await page.reload();
  await page.waitForSelector('.pb-fragment');
  await page.waitForFunction(()=>document.querySelectorAll('.pb-nature-preview canvas[data-ready="true"]').length===4);
  assert(await page.locator('.pb-nature-preview canvas').evaluateAll(nodes=>nodes.every(node=>node.dataset.frame==='0')));
  const reducedFrames=await page.locator('.pb-nature-preview canvas').evaluateAll(nodes=>nodes.map(node=>node.dataset.frame));
  await page.waitForTimeout(200);
  assert.deepEqual(await page.locator('.pb-nature-preview canvas').evaluateAll(nodes=>nodes.map(node=>node.dataset.frame)),reducedFrames);
  for(const width of [390,768]) {
    await page.setViewportSize({width,height:844});
    await page.waitForSelector('.pb-mobile');
    assert.equal(await page.locator('video,canvas,.rs-animation,.nf-animation').count(),0);
    assert(await page.locator('.pb-mobile img').evaluate(n=>n.complete&&n.naturalWidth>0));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  }
  assert.deepEqual(errors,[]);
  console.log('Passed: desktop target visibility, hover menus, four cached moving clips, pause/replay, profile and project views, frozen copy, header order, full-width abstracts, canvas pixels, URL history, Escape, reduced motion, and lightweight mobile. No screenshots.');
} finally {await browser.close();}
