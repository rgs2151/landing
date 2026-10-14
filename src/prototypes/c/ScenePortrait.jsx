import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { profile } from '../../content/site';

export default function ScenePortrait({ selected, slot, canvas, scale, viewport, reduced, close, open, paused, hovered, replay, transitioning }) {
  const root = useRef(null), content = useRef(null), travel = useRef(null), floating = useRef(null);
  const [host, setHost] = useState(null);
  useLayoutEffect(() => { setHost(canvas.current); }, [canvas]);
  useLayoutEffect(() => {
    if (!host) return;
    const keyframes = [[0,0],[3,-1],[1,2],[-2,1],[-3,-2],[0,0]].map(([x,y]) => ({
      transform: `translate(${x}px,${y}px)`, easing: 'cubic-bezier(.45,0,.55,1)',
    }));
    floating.current = content.current.animate(keyframes, { duration: 28000, iterations: Infinity });
    floating.current.pause();
    return () => floating.current.cancel();
  }, [replay, host]);
  useLayoutEffect(() => {
    if (!host) return;
    const node = root.current;
    const sceneBox = host.getBoundingClientRect();
    // Keep the portrait in the scene's stacking order while measuring destinations in viewport space.
    const inScene = box => ({ x: (box.x - sceneBox.x) / scale, y: (box.y - sceneBox.y) / scale, width: box.width / scale, height: box.height / scale });
    const from = node.style.visibility === 'visible' ? inScene(content.current.getBoundingClientRect()) : null;
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
    target = inScene(target);
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
      const box = inScene(slot.current.getBoundingClientRect());
      node.style.top = `${box.y}px`; node.style.left = `${box.x}px`;
    };
    window.addEventListener('scroll', followScroll, { passive: true });
    return () => window.removeEventListener('scroll', followScroll);
  }, [selected, slot, canvas, scale, viewport.width, viewport.height, reduced, host]);
  useEffect(() => {
    if (!host) return;
    if (selected || paused || hovered || reduced || transitioning) floating.current.pause();
    else floating.current.play();
  }, [selected, paused, hovered, reduced, transitioning, replay, scale, viewport.width, viewport.height, host]);
  useEffect(() => () => travel.current?.cancel(), []);
  if (!host) return null;
  return createPortal(<div className="pb-traveling-portrait pc-scene-portrait" ref={root} style={{ visibility: 'hidden' }} data-location={selected === 'about' ? 'about' : selected ? 'stack' : 'home'}>
    <div className="pc-portrait-content" ref={content}>
      <img src={profile.photo} alt={profile.name} />
      {selected !== 'about' && <button disabled={transitioning} onClick={event => selected ? close() : open('about', event.currentTarget)} aria-label={selected ? 'Return to landing page' : `About ${profile.name}`} title={selected ? 'Return to landing page' : 'About'} />}
    </div>
  </div>, host);
}
