import { cloneElement, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { profile } from '../../content/site';

export function useOrbits(root, { desktop, paused, selected, hovered, reduced, replay, diagnostic, transitioning }) {
  const animations = useRef([]);
  useEffect(() => {
    if (!desktop || !root.current) return;
    animations.current = [...root.current.querySelectorAll('.pb-orbit')].map((node, index) => {
      const x = Number(node.dataset.x) - 600, y = Number(node.dataset.y) - 350;
      const phase = Math.atan2(y, x);
      // Nine staggered tracks across the previous radius range, enlarged by 20%.
      const track = (index % 3) * 3 + Math.floor(index / 3);
      const radiusX = (270 + track * 10) * 1.2, radiusY = (160 + track * 6.25) * 1.2;
      const keyframes = Array.from({ length: 121 }, (_, frame) => {
        const angle = phase + frame / 120 * Math.PI * 2;
        return { transform: `translate(${radiusX * Math.cos(angle) - x}px, ${radiusY * Math.sin(angle) - y}px) scale(${.9 + .1 * Math.sin(angle)})` };
      });
      const animation = node.animate(keyframes, { duration: 48000 + index * 2300, iterations: Infinity });
      return { animation, node, settle: null };
    });
    return () => animations.current.forEach(({ animation, settle }) => { animation.cancel(); settle?.cancel(); });
  }, [desktop, root, replay]);
  useEffect(() => {
    animations.current.forEach(({ animation }) => {
      if (paused || selected || reduced || hovered || transitioning || (diagnostic && diagnostic !== 'all' && diagnostic !== 'motion')) animation.pause();
      else animation.play();
    });
  }, [desktop, paused, selected, hovered, reduced, replay, diagnostic, transitioning]);
  useEffect(() => {
    animations.current.forEach(item => {
      const from = getComputedStyle(item.node).transform;
      item.settle?.cancel();
      const to = selected ? 'translate(0px, 0px) scale(1)' : getComputedStyle(item.node).transform;
      item.settle = item.node.animate([{ transform: from }, { transform: to }], {
        duration: reduced ? 0 : 700, easing: 'cubic-bezier(.65,0,.35,1)', fill: 'both',
      });
      if (!selected) item.settle.onfinish = () => item.settle?.cancel();
    });
  }, [desktop, selected, reduced, replay]);
}

// Preserve the accepted frame as inert DOM; discard the running React/WebGL tree.
// SVG and text need no rasterization. Only canvases become small bitmap images.
function cloneFrame(source) {
  const clone = source.cloneNode(true);
  let emptyCanvas = false;
  const originals = source.querySelectorAll('canvas');
  clone.querySelectorAll('canvas').forEach((canvas, index) => {
    const original = originals[index];
    const bitmap = document.createElement('canvas');
    bitmap.width = Math.min(original.width, 160);
    bitmap.height = Math.max(1, Math.round(original.height * bitmap.width / original.width));
    const context = bitmap.getContext('2d');
    context.drawImage(original, 0, 0, bitmap.width, bitmap.height);
    if (!context.getImageData(0, 0, bitmap.width, bitmap.height).data.some((value, i) => i % 4 === 3 && value > 0)) emptyCanvas = true;
    const image = document.createElement('img');
    image.src = bitmap.toDataURL('image/webp', .8);
    image.alt = ''; image.className = 'pb-frozen-canvas';
    canvas.replaceWith(image);
  });
  clone.setAttribute('aria-hidden', 'true');
  return emptyCanvas ? null : clone;
}

export function FreezePreview({ frozen, children }) {
  const live = useRef(null), snapshot = useRef(null);
  const snapshotRef = useRef(null);
  const [captured, setCaptured] = useState(false);
  useLayoutEffect(() => {
    if (!frozen) { setCaptured(false); snapshot.current = null; return; }
    if (captured || !live.current) return;
    let frame, attempts = 0;
    const capture = () => {
      const node = live.current;
      if (!node) return;
      const waiting = node.querySelector('.nf-animation[aria-label], .pb-nature-preview canvas:not([data-ready="true"])');
      if (waiting && attempts++ < 120) { frame = requestAnimationFrame(capture); return; }
      snapshotRef.current?.();
      snapshot.current = cloneFrame(node);
      if (!snapshot.current) { frame = requestAnimationFrame(capture); return; }
      setCaptured(true);
    };
    // Copy inside the paint frame, while the WebGL drawing buffer is available.
    frame = requestAnimationFrame(capture);
    return () => cancelAnimationFrame(frame);
  }, [frozen, captured]);
  return captured && frozen
    ? <div key="frozen" className="pb-frozen" ref={node => { if (node && snapshot.current) node.replaceChildren(snapshot.current); }} />
    : <div key="live" ref={live}>{cloneElement(children, { snapshotRef })}</div>;
}

export function TravelingPortrait({ selected, slot, origin, viewport, reduced, close }) {
  const root = useRef(null), previous = useRef(null);
  useLayoutEffect(() => {
    const node = root.current;
    const from = node.style.visibility === 'visible' ? node.getBoundingClientRect() : null;
    let destination = { x: 72, y: viewport.height - 370, width: 112, height: 140 };
    if (selected === 'about' && slot.current) {
      const box = slot.current.getBoundingClientRect();
      const transform = getComputedStyle(slot.current.closest('.pb-reading')).transform;
      const offset = transform === 'none' ? 0 : new DOMMatrixReadOnly(transform).m41;
      destination = { x: box.x - offset, y: box.y, width: box.width, height: box.height };
    }
    node.getAnimations().forEach(animation => animation.cancel());
    Object.assign(node.style, { left: `${destination.x}px`, top: `${destination.y}px`, width: `${destination.width}px`, height: `${destination.height}px` });
    if (!selected && !previous.current) {
      node.style.visibility = 'hidden';
      return;
    }
    node.style.visibility = 'visible';
    const source = from || (selected === 'about' && origin.current ? {
      x: origin.current.x + origin.current.width / 2 - 56,
      y: origin.current.y + origin.current.height / 2 - 70, width: 112, height: 140,
    } : { ...destination, y: destination.y - 55 });
    const animation = node.animate([
      { transform: `translate(${source.x - destination.x}px,${source.y - destination.y}px) scale(${source.width / destination.width},${source.height / destination.height}) rotate(${from ? 0 : -9}deg)`, opacity: from ? 1 : 0 },
      { transform: 'translate(0,0) scale(1) rotate(0deg)', opacity: selected ? 1 : 0 },
    ], { duration: reduced ? 0 : 650, easing: 'cubic-bezier(.22,.8,.24,1)', fill: 'both' });
    animation.onfinish = () => { if (!selected) node.style.visibility = 'hidden'; };
    previous.current = selected;
    const followScroll = () => {
      if (selected !== 'about' || !slot.current) return;
      const box = slot.current.getBoundingClientRect();
      node.style.top = `${box.y}px`; node.style.left = `${box.x}px`;
    };
    window.addEventListener('scroll', followScroll, { passive: true });
    return () => window.removeEventListener('scroll', followScroll);
  }, [selected, slot, origin, viewport.width, viewport.height, reduced]);
  return <div className="pb-traveling-portrait" ref={root} style={{ visibility: 'hidden' }} aria-hidden={!selected}>
    <img src={profile.photo} alt={profile.name} />
    {selected && selected !== 'about' && <button onClick={close} aria-label="Return to landing page" title="Return to landing page" />}
  </div>;
}
