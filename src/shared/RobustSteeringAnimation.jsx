import { useEffect, useMemo, useRef } from 'react';
import sourcePage from '../../design/references/robust-steering-animation/source-page.html?raw';
import sourceStyles from '../../design/references/robust-steering-animation/steering-demo.css?raw';
import sourceScript from '../../design/references/robust-steering-animation/steering-demo.js?raw';
import './robust-steering-animation.css';

function animationDocument(compact) {
  const document = new DOMParser().parseFromString(sourcePage, 'text/html');
  const animation = document.querySelector('.steering-demo');
  // Run the original code in an isolated, fixed desktop viewport so its CSS,
  // SVG IDs and measurements work unchanged at both display sizes.
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    * { box-sizing: border-box; }
    html, body { margin: 0; width: 1000px; height: 550px; overflow: hidden; }
    body { --muted: #666; --teal: #007c7c; background: white; }
    ${sourceStyles}
    .steering-demo { width: 776px; margin: 0; height: 550px; }
    .steering-demo-layout { height: 436px; }
    .steering-demo-controls { ${compact ? 'visibility: hidden;' : ''} }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
  </style></head><body>${animation.outerHTML}<script>${sourceScript}</script><script>
    const box = document.querySelector('.steering-demo');
    const pause = box.querySelector('.steering-demo-pause');
    let suspended = false;
    let wasRunning = false;
    window.addEventListener('message', event => {
      if (event.source !== parent || event.data?.type !== 'steering-active') return;
      if (!event.data.active && !suspended) {
        wasRunning = box.dataset.paused !== 'true';
        if (wasRunning) pause.click();
        suspended = true;
      } else if (event.data.active && suspended) {
        if (wasRunning && box.dataset.paused === 'true') pause.click();
        suspended = false;
      }
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') parent.postMessage({type: 'steering-close'}, '*');
    });
  </script></body></html>`;
}

export default function RobustSteeringAnimation({ compact = false, active = true, onClose }) {
  const root = useRef(null);
  const frame = useRef(null);
  const srcDoc = useMemo(() => animationDocument(compact), [compact]);
  const sync = () => frame.current?.contentWindow?.postMessage({ type: 'steering-active', active }, '*');

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
      if (event.source === frame.current?.contentWindow && event.data?.type === 'steering-close') onClose?.();
    };
    window.addEventListener('message', close);
    return () => window.removeEventListener('message', close);
  }, [onClose]);

  return <div className={`rs-animation${compact ? ' rs-compact' : ''}`} ref={root}>
    <iframe ref={frame} title="Robust steering animation" srcDoc={srcDoc}
      sandbox="allow-scripts" tabIndex={compact ? -1 : 0} aria-hidden={compact || undefined}
      onLoad={sync} />
  </div>;
}
