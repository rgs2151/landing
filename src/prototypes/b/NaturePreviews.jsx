import { createContext, useContext, useEffect, useRef } from 'react';
import clips from '../../../public/assets/images/previews/clips.json';

const Clock = createContext(null);
const sheets = new Map();
function loadSheet(src) {
  if (!sheets.has(src)) sheets.set(src, new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => { sheets.delete(src); reject(new Error(`Preview unavailable: ${src}`)); };
    image.src = src;
  }));
  return sheets.get(src);
}

// One paint clock, no per-frame React renders or continuous video decoders.
export function NaturePreviewClock({ active, replay, children }) {
  const clock = useRef({ elapsed: 0, targets: new Set() });
  useEffect(() => {
    clock.current.elapsed = 0;
    clock.current.targets.forEach(draw => draw(0));
  }, [replay]);
  useEffect(() => {
    let frame, previous;
    const tick = now => {
      if (previous !== undefined) clock.current.elapsed += Math.min(now - previous, 100);
      previous = now;
      clock.current.targets.forEach(draw => draw(clock.current.elapsed));
      frame = requestAnimationFrame(tick);
    };
    const sync = () => {
      cancelAnimationFrame(frame); previous = undefined;
      if (active && !document.hidden) frame = requestAnimationFrame(tick);
    };
    sync(); document.addEventListener('visibilitychange', sync);
    return () => { cancelAnimationFrame(frame); document.removeEventListener('visibilitychange', sync); };
  }, [active]);
  return <Clock.Provider value={clock.current}>{children}</Clock.Provider>;
}

export default function NaturePreview({ clip, poster }) {
  const canvas = useRef(null);
  const clock = useContext(Clock);
  useEffect(() => {
    const node = canvas.current, context = node.getContext('2d', { alpha: false });
    const data = clips[clip];
    let disposed = false, draw, visible = false;
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; });
    observer.observe(node);
    loadSheet(`${data.src}?v=${data.revision}`).then(sheet => {
      if (disposed) return;
      if (sheet.naturalWidth !== data.columns * data.size || sheet.naturalHeight !== Math.ceil(data.frames / data.columns) * data.size) {
        throw new Error(`Preview dimensions do not match metadata: ${data.src}`);
      }
      let previous = -1;
      draw = elapsed => {
        const frame = Math.floor(elapsed * data.fps / 1000) % data.frames;
        if (frame === previous || (!visible && previous !== -1)) return;
        context.drawImage(sheet, frame % data.columns * data.size, Math.floor(frame / data.columns) * data.size,
          data.size, data.size, 0, 0, data.size, data.size);
        node.dataset.frame = frame;
        node.dataset.ready = 'true';
        previous = frame;
      };
      draw(clock.elapsed); clock.targets.add(draw);
    }).catch(() => { if (!disposed) node.dataset.failed = 'true'; });
    return () => { disposed = true; observer.disconnect(); clock.targets.delete(draw); };
  }, [clip, clock]);
  return <div className="pb-nature-preview">
    <img src={poster} alt="" />
    <canvas ref={canvas} width={clips[clip].size} height={clips[clip].size} aria-hidden="true" />
  </div>;
}
