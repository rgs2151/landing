import SVGAnimation from '../SVGAnimation';
import clips from '../../../public/assets/images/previews/neuraljaxlib/clips.json';

async function loadBitmap(src) {
  const response = await fetch(src);
  if (!response.ok) throw new Error(`Cached preview unavailable: ${src}`);
  return createImageBitmap(await response.blob());
}

function renderFrames(root, { clip, first }) {
  const heading = document.createElement('div');
  heading.textContent = clip.heading;
  heading.setAttribute('style', clip.headingStyle);
  const canvas = document.createElement('canvas');
  canvas.width = clip.width; canvas.height = clip.height;
  canvas.setAttribute('style', clip.svgStyle);
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', clip.label || clip.heading);
  if (clip.width === 57) canvas.style.imageRendering = 'pixelated';
  const context = canvas.getContext('2d');
  const images = new Map([[0, { bitmap: first }]]);
  let disposed = false, elapsed = 0, last = -1;
  root.dataset.cached = 'true';
  root.append(heading, canvas);

  function request(index) {
    if (images.has(index)) return;
    const entry = {};
    images.set(index, entry);
    loadBitmap(clip.sheets[index].src).then(bitmap => {
      if (disposed || images.get(index) !== entry) { bitmap.close(); return; }
      entry.bitmap = bitmap;
      draw();
    }).catch(() => {
      images.delete(index);
      root.dataset.cacheFailed = 'true';
    });
  }
  function draw() {
    if (disposed) return;
    const frame = Math.floor((elapsed % clip.duration) / clip.duration * clip.count);
    const unique = clip.timeline[frame];
    const index = clip.sheets.findIndex(sheet => unique >= sheet.start && unique < sheet.start + sheet.count);
    const next = (index + 1) % clip.sheets.length;
    request(index); request(next);
    // Keep the current and upcoming sheet, plus the shared first-frame sheet.
    for (const [key, value] of images) {
      if (key !== 0 && key !== index && key !== next) { value.bitmap?.close(); images.delete(key); }
    }
    const image = images.get(index)?.bitmap;
    if (!image || last === unique) return;
    const sheet = clip.sheets[index], local = unique - sheet.start;
    context.clearRect(0, 0, clip.width, clip.height);
    context.drawImage(image, local % sheet.columns * clip.width, Math.floor(local / sheet.columns) * clip.height,
      clip.width, clip.height, 0, 0, clip.width, clip.height);
    root.dataset.frame = String(frame);
    canvas.dataset.ready = 'true';
    last = unique;
  }
  draw();
  return {
    duration: clip.duration,
    seek(ms) { elapsed = Math.max(0, ms); draw(); },
    pause() {},
    destroy() {
      disposed = true;
      images.forEach(({ bitmap }, index) => { if (index !== 0) bitmap?.close(); });
      images.clear(); heading.remove(); canvas.remove();
    },
  };
}

export const cachedNeuraljaxPreviews = Object.fromEntries(Object.entries(clips).map(([id, clip]) => {
  let loaded;
  const asset = {
    id: `cached-${id}`, label: clip.heading, fps: clip.count * 1000 / clip.duration,
    load: () => loaded ??= loadBitmap(clip.sheets[0].src).then(first => ({ render: renderFrames, data: { clip, first } }))
      .catch(error => { loaded = null; throw error; }),
  };
  return [`neuraljax-${id}`, props => <SVGAnimation {...props} asset={asset} />];
}));
