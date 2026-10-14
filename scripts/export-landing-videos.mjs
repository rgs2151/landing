import { mkdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createRequire } from 'node:module';
import playwright from '/tmp/landing-browser-check/node_modules/playwright/index.js';
import { previewSources } from './preview-sources.mjs';
const sharp = createRequire('/tmp/landing-browser-check/package.json')('sharp');

const browser = await playwright.chromium.launch({
  executablePath: '/home/dev/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',
  args: ['--enable-unsafe-swiftshader'],
});
const selected = process.env.ASSETS?.split(',');
const verifyOnly = process.env.VERIFY_ONLY === '1';
const manifests = new Map();
let runningEncoder;
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
  // Writing preview manifests must not hot-reload the active capture document.
  await page.routeWebSocket('**', () => {});
  await page.goto('http://localhost:5173/?benchmark=idle');
  await page.waitForSelector('.pb-frozen');
  const entries = await page.evaluate(async () => {
    const [react, dom, client, catalog] = await Promise.all([
      import('/node_modules/.vite/deps/react.js'), import('/node_modules/.vite/deps/react-dom.js'),
      import('/node_modules/.vite/deps/react-dom_client.js'), import('/src/assets/exportCatalog.js'),
    ]);
    const host = document.createElement('div');
    host.className = 'site'; host.style.cssText = 'position:fixed;left:-10000px;top:0;min-height:0;background:transparent;overflow:visible;';
    const mount = document.createElement('div'); host.append(mount); document.body.append(host);
    window.exporter = { React: react.default || react, flushSync: (dom.default || dom).flushSync,
      root: (client.default || client).createRoot(mount), mount, host, entries: await catalog.exportCatalog() };
    return window.exporter.entries.map(({ Component, ...entry }) => entry);
  });
  for (const entry of entries.filter(e => !selected || selected.includes(e.id))) {
    const base = `public/assets/projects/${entry.project}`;
    mkdirSync(`${base}/videos`, { recursive: true }); mkdirSync(`${base}/posters`, { recursive: true });
    if (!manifests.has(entry.project)) {
      let previous = {}; try { previous = JSON.parse(readFileSync(`${base}/previews.json`)); } catch {}
      manifests.set(entry.project, previous);
    }
    const layout = await page.evaluate(async id => {
      const e = window.exporter;
      e.entry = e.entries.find(item => item.id === id);
      e.host.style.width = `${e.entry.sourceWidth}px`;
      e.render = ms => e.flushSync(() => e.root.render(e.React.createElement(e.entry.Component, { compact: true, active: false, elapsedMs: ms, ...e.entry.props })));
      e.render(0);
      const start = performance.now();
      while (e.mount.querySelector('[data-ready="false"], .nf-animation[aria-label]') || !e.mount.firstElementChild) {
        if (performance.now() - start > 15000) throw new Error(`Source failed to load: ${id}`);
        await new Promise(resolve => setTimeout(resolve, 30));
      }
      await new Promise(resolve => setTimeout(resolve, 100));
      e.render(0);
      const bounds = e.mount.firstElementChild.getBoundingClientRect();
      e.layout = { width: bounds.width, height: bounds.height };
      e.css = [...document.styleSheets].flatMap(sheet => {
        try { return [...sheet.cssRules].filter(rule => !rule.cssText.startsWith('@font-face')).map(rule => rule.cssText); } catch { return []; }
      }).join('\n');
      return e.layout;
    }, entry.id);
    const width = Math.ceil(entry.width / 2) * 2;
    const height = Math.ceil(width * layout.height / layout.width / 2) * 2;
    const count = Math.ceil(entry.duration * entry.fps / 1000);
    const fps = count * 1000 / entry.duration;
    const file = `${base}/videos/${entry.name}.webm`;
    const encoder = verifyOnly ? null : spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-framerate', String(fps), '-i', 'pipe:0',
      '-an', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-auto-alt-ref', '0', '-b:v', '0', '-crf', String(entry.crf ?? 15), '-deadline', 'good', '-cpu-used', '4', '-row-mt', '1', '-threads', '2', file]);
    runningEncoder = encoder;
    let stderr = ''; encoder?.stderr.on('data', data => { stderr += data; });
    const finished = encoder ? once(encoder, 'close') : null;
    let firstStats, lastHash;
    for (let frame = 0; frame < (verifyOnly ? 1 : count); frame++) {
      const rendered = await page.evaluate(async ({ ms, width, height }) => {
        const e = window.exporter;
        e.render(ms);
        // Allow seek effects to settle without advancing the animation's own clock.
        await Promise.resolve();
        for (const animation of e.mount.getAnimations({ subtree: true })) { animation.pause(); animation.currentTime = ms; }
        const source = e.mount.firstElementChild;
        const copy = source.cloneNode(true);
        const liveCanvases = source.querySelectorAll('canvas');
        copy.querySelectorAll('canvas').forEach((node, i) => {
          const image = document.createElement('img'); image.src = liveCanvases[i].toDataURL();
          image.style.cssText = 'display:block;width:100%;height:100%'; node.replaceWith(image);
        });
        const liveAnimations = source.querySelectorAll('animate');
        copy.querySelectorAll('animate').forEach((node, i) => {
          node.parentElement.style.opacity = getComputedStyle(liveAnimations[i].parentElement).opacity;
          node.remove();
        });
        const bolt = source.querySelector('.steering-demo-bolt');
        if (bolt) copy.querySelector('.steering-demo-bolt').style.transform = getComputedStyle(bolt).transform;
        const wrap = document.createElement('div');
        wrap.className = 'site';
        wrap.style.cssText = `width:${e.layout.width}px;height:${e.layout.height}px;min-height:0;background:transparent;overflow:hidden;`;
        const style = document.createElement('style'); style.textContent = `${e.css}\n:root{background:transparent!important}*{animation:none!important;transition:none!important}`;
        wrap.append(style, copy);
        const html = new XMLSerializer().serializeToString(wrap);
        const xml = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${e.layout.width} ${e.layout.height}"><foreignObject width="100%" height="100%">${html}</foreignObject></svg>`;
        const image = new Image(); image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(xml)}`; await image.decode();
        const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
        const context = canvas.getContext('2d'); context.drawImage(image, 0, 0);
        const rgba = context.getImageData(0, 0, width, height).data;
        let visible = 0, transparent = 0;
        for (let i = 3; i < rgba.length; i += 4) { if (rgba[i]) visible++; if (rgba[i] === 0) transparent++; }
        return { png: canvas.toDataURL('image/png').split(',')[1], visible, transparent };
      }, { ms: frame * entry.duration / count, width, height });
      const png = Buffer.from(rendered.png, 'base64');
      if (!rendered.visible) throw new Error(`Blank asset frame: ${entry.id}/${frame}`);
      lastHash = createHash('sha256').update(png).digest('hex');
      if (verifyOnly) {
        const previous = manifests.get(entry.project)[entry.id];
        if (lastHash !== previous?.firstFrame.sha256) {
          const expected = await sharp(`public${previous.poster}`).ensureAlpha().raw().toBuffer();
          const actual = await sharp(png).ensureAlpha().raw().toBuffer();
          let maxDifference = 0, changedPixels = 0;
          if (actual.length !== expected.length) throw new Error(`Source frame dimensions changed: ${entry.id}`);
          for (let i = 0; i < actual.length; i += 4) {
            let difference = Math.abs(actual[i + 3] - expected[i + 3]);
            for (let channel = 0; channel < 3; channel++) {
              const a = actual[i + channel] * actual[i + 3] / 255 + 255 - actual[i + 3];
              const b = expected[i + channel] * expected[i + 3] / 255 + 255 - expected[i + 3];
              difference = Math.max(difference, Math.abs(a - b));
            }
            maxDifference = Math.max(maxDifference, difference);
            if (difference > 1) changedPixels++;
          }
          if (maxDifference > 1) throw new Error(`Source first frame changed: ${entry.id}; max difference ${maxDifference}, ${changedPixels} pixels`);
        }
        console.log(`${entry.id}: original first-frame pixels unchanged`);
        continue;
      }
      if (!frame) {
        firstStats = { visible: rendered.visible, transparent: rendered.transparent, sha256: lastHash };
        await sharp(png).webp({ lossless: true }).toFile(`${base}/posters/${entry.name}.webp`);
      }
      if (!encoder.stdin.write(png)) await once(encoder.stdin, 'drain');
      if (frame && frame % 300 === 0) console.log(`${entry.id}: ${frame}/${count} frames`);
    }
    if (verifyOnly) continue;
    encoder.stdin.end();
    const [code] = await finished;
    runningEncoder = null;
    if (code) throw new Error(`Encoding failed: ${stderr}`);
    const revision = createHash('sha256').update(readFileSync(file)).digest('hex').slice(0, 12);
    const item = { id: entry.id, project: entry.project, name: entry.name, src: `/assets/projects/${entry.project}/videos/${entry.name}.webm`,
      poster: `/assets/projects/${entry.project}/posters/${entry.name}.webp`, revision, width, height,
      layoutWidth: layout.width, layoutHeight: layout.height, duration: entry.duration, loopStart: entry.loopStart || 0,
      fps, frames: count, crf: entry.crf ?? 15, exportPixelRatio: entry.props?.exportPixelRatio || null,
      hideTitle: entry.props?.hideTitle || false,
      bytes: statSync(file).size, firstFrame: firstStats, lastFrameHash: lastHash,
      fragment: `src/assets/${entry.project}/fragments.js#${entry.name}`, composite: `src/assets/${entry.project}/composite.js`, sources: previewSources(entry) };
    manifests.get(entry.project)[entry.id] = item;
    writeFileSync(`${base}/previews.json`, JSON.stringify(manifests.get(entry.project), null, 2) + '\n');
    console.log(`${entry.id}: saved ${width}x${height}, ${(entry.duration / 1000).toFixed(2)}s, ${item.bytes} bytes`);
  }
} finally {
  if (runningEncoder && runningEncoder.exitCode === null) {
    runningEncoder.stdin.destroy(); runningEncoder.kill(); await once(runningEncoder, 'close');
  }
  await browser.close();
}
