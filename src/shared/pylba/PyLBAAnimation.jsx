import { createElement, useEffect, useId, useMemo, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import source from './pylba.svg?raw';
import './styles.css';

let template;
function svgContent(prefix) {
  template ??= new DOMParser().parseFromString(source, 'image/svg+xml').documentElement;
  const render = (node, key) => {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node.nodeType !== Node.ELEMENT_NODE || node.localName === 'style') return null;
    // Preserve the supplied geometry; remove only its page-sized white backing.
    if (node.parentNode === template && node.localName === 'rect' && node.getAttribute('width') === '300') return null;
    const props = { key };
    for (const attribute of node.attributes) {
      const name = attribute.name === 'class' ? 'className'
        : attribute.name.startsWith('aria-') ? attribute.name
        : attribute.name.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      props[name] = attribute.name === 'id' ? `${prefix}-${attribute.value}`
        : attribute.value.replace(/url\(#([^)]+)\)/g, (_, id) => `url(#${prefix}-${id})`);
    }
    return createElement(node.localName, props, ...Array.from(node.childNodes, render));
  };
  return Array.from(template.childNodes, render);
}

export default function PyLBAAnimation({ compact = false, active = true, snapshotRef, elapsedMs }) {
  const prefix = `pylba-${useId().replaceAll(':', '')}`;
  const content = useMemo(() => svgContent(prefix), [prefix]);
  const svg = useRef(null);
  const root = useRef(null);
  const [paused, setPaused] = useState(false);
  const [reduced, setReduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [visible, setVisible] = useState(false);
  const [foreground, setForeground] = useState(() => !document.hidden);
  const playing = active && visible && foreground && !paused && !reduced && elapsedMs === undefined;

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    observer.observe(root.current);
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const motionChange = () => setReduced(media.matches);
    const visibilityChange = () => setForeground(!document.hidden);
    media.addEventListener('change', motionChange);
    document.addEventListener('visibilitychange', visibilityChange);
    return () => {
      observer.disconnect();
      media.removeEventListener('change', motionChange);
      document.removeEventListener('visibilitychange', visibilityChange);
    };
  }, []);
  useEffect(() => {
    if (playing) svg.current.unpauseAnimations();
    else svg.current.pauseAnimations();
  }, [playing]);
  useEffect(() => { if (elapsedMs !== undefined) svg.current.setCurrentTime(elapsedMs / 1000); }, [elapsedMs]);
  useEffect(() => {
    if (!snapshotRef) return;
    // FreezePreview clones inert SVG: capture animated opacity before removing SMIL.
    snapshotRef.current = () => {
      svg.current.pauseAnimations();
      svg.current.querySelectorAll('animate').forEach(animation => {
        const path = animation.parentElement;
        path.setAttribute('opacity', getComputedStyle(path).opacity);
        animation.remove();
      });
    };
    return () => { snapshotRef.current = null; };
  }, [snapshotRef]);

  return <div className={`pylba-animation${compact ? ' pylba-compact' : ''}`} ref={root} data-reduced={reduced}>
    <svg ref={svg} viewBox={`0 0 300 ${compact ? 300 : 338}`} role="img" aria-labelledby={`${prefix}-description`}>
      {content}
    </svg>
    {!compact && <div className="pylba-controls">
      <button type="button" title={paused || reduced ? 'Play' : 'Pause'} aria-label={`${paused || reduced ? 'Play' : 'Pause'} PyLBA animation`}
        onClick={() => { if (reduced) { setReduced(false); setPaused(false); } else setPaused(value => !value); }}>
        {paused || reduced ? <Play size={17} /> : <Pause size={17} />}
      </button>
      <button type="button" title="Replay" aria-label="Replay PyLBA animation" onClick={() => svg.current.setCurrentTime(0)}><RotateCcw size={17} /></button>
    </div>}
  </div>;
}
