import { useEffect, useLayoutEffect, useRef } from 'react';
import { profile } from '../../content/site';

export default function ScenePortrait({ selected, slot, canvas, scale, viewport, reduced, close, paused, hovered, replay, transitioning }) {
  const root = useRef(null), content = useRef(null), travel = useRef(null), floating = useRef(null);
  useLayoutEffect(() => {
    const keyframes = [[0,0],[3,-1],[1,2],[-2,1],[-3,-2],[0,0]].map(([x,y]) => ({
      transform: `translate(${x}px,${y}px)`, easing: 'cubic-bezier(.45,0,.55,1)',
    }));
    floating.current = content.current.animate(keyframes, { duration: 28000, iterations: Infinity });
    floating.current.pause();
    return () => floating.current.cancel();
  }, [replay]);
  useLayoutEffect(() => {
    const node = root.current;
    const from = node.style.visibility === 'visible' ? content.current.getBoundingClientRect() : null;
    let target;
    if (selected === 'about' && slot.current) {
      const box = slot.current.getBoundingClientRect();
      const transform = getComputedStyle(slot.current.closest('.pb-reading')).transform;
      const offset = transform === 'none' ? 0 : new DOMMatrixReadOnly(transform).m41;
      target = { x: box.x - offset, y: box.y, width: box.width, height: box.height };
    } else if (selected) {
      target = { x: 72, y: viewport.height - 370, width: 112, height: 112 };
    } else {
      const box = canvas.current.getBoundingClientRect();
      const title = canvas.current.querySelector('.pb-title');
      const right = Math.max(...Array.from(title.children, line => line.offsetLeft + line.offsetWidth));
      const size = title.offsetHeight * scale;
      target = { x: box.x + (title.offsetLeft + right + 20) * scale, y: box.y + title.offsetTop * scale, width: size, height: size };
    }
    travel.current?.cancel();
    floating.current.pause();
    floating.current.currentTime = 0;
    Object.assign(node.style, { left: `${target.x}px`, top: `${target.y}px`, width: `${target.width}px`, height: `${target.height}px`, visibility: 'visible' });
    if (from) {
      travel.current = node.animate([
        { transform: `translate(${from.x-target.x}px,${from.y-target.y}px) scale(${from.width/target.width},${from.height/target.height})` },
        { transform: 'translate(0,0) scale(1)' },
      ], { duration: reduced ? 0 : 650, easing: 'cubic-bezier(.22,.8,.24,1)', fill: 'both' });
    }
    const followScroll = () => {
      if (selected !== 'about' || !slot.current) return;
      const box = slot.current.getBoundingClientRect();
      node.style.top = `${box.y}px`; node.style.left = `${box.x}px`;
    };
    window.addEventListener('scroll', followScroll, { passive: true });
    return () => window.removeEventListener('scroll', followScroll);
  }, [selected, slot, canvas, scale, viewport.width, viewport.height, reduced]);
  useEffect(() => {
    if (selected || paused || hovered || reduced || transitioning) floating.current.pause();
    else floating.current.play();
  }, [selected, paused, hovered, reduced, transitioning, replay, scale, viewport.width, viewport.height]);
  useEffect(() => () => travel.current?.cancel(), []);
  return <div className="pb-traveling-portrait pc-scene-portrait" ref={root} style={{ visibility: 'hidden' }} data-location={selected === 'about' ? 'about' : selected ? 'stack' : 'home'}>
    <div className="pc-portrait-content" ref={content}>
      <img src={profile.photo} alt={profile.name} />
      {selected && selected !== 'about' && <button onClick={close} aria-label="Return to landing page" title="Return to landing page" />}
    </div>
  </div>;
}
