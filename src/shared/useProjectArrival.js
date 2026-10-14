import { useLayoutEffect, useState } from 'react';

export default function useProjectArrival({ enabled, selected, desktop, canvas, scale, width, height }) {
  const [target, setTarget] = useState(null);
  useLayoutEffect(() => {
    if (!enabled || !desktop || !selected || selected === 'about' || !canvas.current) return;
    const reading = canvas.current.parentElement.querySelector('.pb-reading');
    if (!reading) return;
    const figure = reading.querySelector('.pb-project-media, .pb-tool-gallery, .pb-tool-media') || reading;
    const update = () => {
      const box = figure.getBoundingClientRect(), scene = canvas.current.getBoundingClientRect();
      // The reading page slides in from the right; target its final position.
      const transform = getComputedStyle(reading).transform;
      const offset = transform === 'none' ? 0 : new DOMMatrixReadOnly(transform).m41;
      const x = (box.left + box.width / 2 - offset - scene.left) / scale;
      const y = (Math.min(height - 80, Math.max(120, box.top + box.height / 2)) - scene.top) / scale;
      const heights = Object.fromEntries([...canvas.current.querySelectorAll('.pb-fragment')]
        .filter(node => node.dataset.project === selected)
        .map(node => [node.dataset.piece, node.offsetHeight]));
      const next = { project: selected, x, y, heights };
      setTarget(current => JSON.stringify(current) === JSON.stringify(next) ? current : next);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(figure);
    for (const node of canvas.current.querySelectorAll('.pb-fragment')) {
      if (node.dataset.project === selected) observer.observe(node);
    }
    const frame = requestAnimationFrame(update);
    return () => { observer.disconnect(); cancelAnimationFrame(frame); };
  }, [enabled, selected, desktop, canvas, scale, width, height]);
  return target;
}
