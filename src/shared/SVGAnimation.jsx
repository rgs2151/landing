import { useEffect, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import './svg-animation.css';

// React owns the lifecycle; the supplied renderer owns only its SVG subtree.
export default function SVGAnimation({ asset, compact = false, active = true, snapshotRef, elapsedMs }) {
  const root = useRef(null), mount = useRef(null), controller = useRef(null);
  const elapsed = useRef(0), frame = useRef(0);
  const [ready, setReady] = useState(false), [failed, setFailed] = useState(false);
  const [paused, setPaused] = useState(false), [visible, setVisible] = useState(false);
  const [foreground, setForeground] = useState(() => !document.hidden);
  const [reduced, setReduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [motionOverride, setMotionOverride] = useState(false);
  const [dimensions, setDimensions] = useState({ scale: 1, height: 300 });
  const playing = ready && active && visible && foreground && !paused && (!reduced || motionOverride) && elapsedMs === undefined;

  useEffect(() => {
    let disposed = false;
    asset.load().then(({ render, data }) => {
      if (disposed || !mount.current) return;
      controller.current = render(mount.current, data);
      controller.current.pause();
      // Remove page backings, not meaningful white regions within the artwork.
      const svg = mount.current.querySelector('svg');
      if (getComputedStyle(svg).backgroundColor === 'rgb(255, 255, 255)') svg.style.background = 'transparent';
      setReady(true);
    }).catch(error => { if (!disposed) { setFailed(true); console.error(error); } });
    return () => { disposed = true; cancelAnimationFrame(frame.current); controller.current?.destroy(); controller.current = null; };
  }, [asset]);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    observer.observe(root.current);
    const resize = new ResizeObserver(() => {
      if (!root.current || !mount.current) return;
      const scale = root.current.clientWidth / 300;
      setDimensions({ scale, height: mount.current.offsetHeight * scale });
    });
    resize.observe(root.current); resize.observe(mount.current);
    const preference = matchMedia('(prefers-reduced-motion: reduce)');
    const motionChange = () => { setReduced(preference.matches); setMotionOverride(false); };
    const visibilityChange = () => setForeground(!document.hidden);
    preference.addEventListener('change', motionChange);
    document.addEventListener('visibilitychange', visibilityChange);
    return () => {
      observer.disconnect(); resize.disconnect();
      preference.removeEventListener('change', motionChange);
      document.removeEventListener('visibilitychange', visibilityChange);
    };
  }, []);

  useEffect(() => {
    if (!ready || elapsedMs === undefined) return;
    elapsed.current = elapsedMs;
    controller.current.seek(elapsedMs);
  }, [ready, elapsedMs]);

  useEffect(() => {
    if (!playing) return;
    let last = null, drawn = -Infinity;
    const tick = now => {
      if (!root.current || !controller.current) return;
      if (last !== null) elapsed.current += Math.min(now - last, 100);
      last = now;
      if (now - drawn >= 1000 / asset.fps) {
        controller.current.seek(elapsed.current);
        root.current.dataset.elapsed = String(elapsed.current);
        drawn = now;
      }
      frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame.current);
  }, [playing, asset]);

  useEffect(() => {
    if (!snapshotRef) return;
    snapshotRef.current = () => {
      cancelAnimationFrame(frame.current); controller.current?.pause();
      if (root.current) root.current.dataset.playing = 'false';
    };
    return () => { snapshotRef.current = null; };
  }, [snapshotRef]);

  const stopped = paused || (reduced && !motionOverride);
  return <div className="svg-asset" ref={root} data-asset={asset.id} data-ready={ready} data-playing={playing}
    style={{ height: dimensions.height + (compact ? 0 : 40) }}>
    <div className="svg-asset-mount" ref={mount} style={{ transform: `scale(${dimensions.scale})` }} />
    {failed && <span role="alert">Animation unavailable</span>}
    {!compact && <div className="svg-asset-controls">
      <button type="button" disabled={!ready} title={stopped ? 'Play' : 'Pause'} aria-label={`${stopped ? 'Play' : 'Pause'} ${asset.label} animation`}
        onClick={() => { if (stopped) { setPaused(false); setMotionOverride(true); } else setPaused(true); }}>
        {stopped ? <Play size={17} /> : <Pause size={17} />}
      </button>
      <button type="button" disabled={!ready} title="Replay" aria-label={`Replay ${asset.label} animation`} onClick={() => {
        elapsed.current = 0; controller.current.seek(0); root.current.dataset.elapsed = '0';
      }}><RotateCcw size={17} /></button>
    </div>}
  </div>;
}
