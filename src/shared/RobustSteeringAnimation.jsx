import { useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import sourcePage from '../../design/references/robust-steering-animation/source-page.html?raw';
import sourceStyles from '../../design/references/robust-steering-animation/steering-demo.css?raw';
import sourceScript from '../../design/references/robust-steering-animation/steering-demo.js?raw';
import './robust-steering-animation.css';

function animationDocument() {
  const document = new DOMParser().parseFromString(sourcePage, 'text/html');
  const animation = document.querySelector('.steering-demo');
  // Isolate the source SVG IDs and desktop measurements at both display sizes.
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 1000px; height: 550px; overflow: hidden; }
    body { --muted: #666; --teal: #007c7c; background: transparent; }
    ${sourceStyles}
    .steering-demo { width: 776px; margin: 0; height: 550px; background: transparent; }
    .steering-demo-layout { height: 436px; }
    .steering-demo-controls { display: none; }
    .steering-demo-state, .steering-demo-scene { opacity: 1; }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
  </style></head><body>${animation.outerHTML}<script>${sourceScript}</script><script>
    const box = document.querySelector('.steering-demo');
    const pause = box.querySelector('.steering-demo-pause');
    let suspended = false;
    let wasRunning = false;
    const report = () => parent.postMessage({ type: 'steering-state', paused: box.dataset.paused === 'true' }, '*');
    new MutationObserver(report).observe(box, { attributes: true, attributeFilter: ['data-paused'] });
    window.addEventListener('message', event => {
      if (event.source !== parent) return;
      if (event.data?.type === 'steering-control') {
        if (event.data.command === 'toggle') pause.click();
        if (event.data.command === 'replay') box.steeringDemo.replay();
        report();
        return;
      }
      if (event.data?.type !== 'steering-active') return;
      if (!event.data.active && !suspended) {
        wasRunning = box.dataset.paused !== 'true';
        if (wasRunning) pause.click();
        suspended = true;
      } else if (event.data.active && suspended) {
        if (wasRunning && box.dataset.paused === 'true') pause.click();
        suspended = false;
      }
      report();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') parent.postMessage({type: 'steering-close'}, '*');
    });
  </script></body></html>`;
}

export default function RobustSteeringAnimation({ compact = false, active = true, onClose }) {
  const root = useRef(null);
  const frame = useRef(null);
  const [paused, setPaused] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const srcDoc = useMemo(animationDocument, []);
  const sync = () => frame.current?.contentWindow?.postMessage({ type: 'steering-active', active }, '*');
  const command = value => frame.current?.contentWindow?.postMessage({ type: 'steering-control', command: value }, '*');

  useEffect(() => {
    const resize = new ResizeObserver(([entry]) => {
      const scale = Math.min(entry.contentRect.width / 776, entry.contentRect.height / 550);
      root.current.style.setProperty('--steering-scale', scale);
    });
    resize.observe(root.current);
    return () => resize.disconnect();
  }, []);
  useEffect(sync, [active]);
  useEffect(() => {
    const close = event => {
      if (event.source !== frame.current?.contentWindow) return;
      if (event.data?.type === 'steering-close') onClose?.();
      if (event.data?.type === 'steering-state') setPaused(event.data.paused);
    };
    window.addEventListener('message', close);
    return () => window.removeEventListener('message', close);
  }, [onClose]);

  return <div className={`rs-animation${compact ? ' rs-compact' : ''}`} ref={root}>
    <iframe ref={frame} title="Robust steering animation" srcDoc={srcDoc}
      sandbox="allow-scripts" tabIndex={compact ? -1 : 0} aria-hidden={compact || undefined}
      onLoad={sync} />
    {!compact && <div className="rs-controls">
      <button type="button" onClick={() => command('toggle')} aria-label={paused ? 'Play robust steering animation' : 'Pause robust steering animation'} title={paused ? 'Play' : 'Pause'}>{paused ? <Play size={17} /> : <Pause size={17} />}</button>
      <button type="button" onClick={() => command('replay')} aria-label="Replay robust steering animation" title="Replay"><RotateCcw size={17} /></button>
    </div>}
  </div>;
}
