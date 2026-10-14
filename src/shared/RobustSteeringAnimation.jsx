import { memo, useEffect, useId, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import landscapes from '../../design/references/robust-steering-animation/steering-landscapes.json';
import { ease, sampleSteering } from './robustSteeringModel';
import './robust-steering-animation.css';

const fields = Object.fromEntries(landscapes.fields.map(field => [field.language, field]));
const Field = memo(function Field({ language }) {
  return fields[language].paths.map((path, i) => <path key={i} d={path.d} fill={path.fill}
    fillOpacity={path['fill-opacity']} fillRule={path['fill-rule']} stroke={path.stroke}
    strokeOpacity={path['stroke-opacity']} strokeWidth={path['stroke-width']} strokeLinejoin={path['stroke-linejoin']} />);
});

function Landscape({ snapshot }) {
  const id = useId().replaceAll(':', '');
  const { entry, previous, first, swipe, drawn, stored, directionOpacity, labelsOpacity, phase } = snapshot;
  const prior = first ? entry : previous;
  const route = fields[entry.language].route;
  const scale = 1 - stored * .94;
  // Accepted lightning center in the landscape's SVG coordinates.
  const target = { x: (386 - 20) * 340 / 320, y: 10 + (274.78125 - 44 * 36 / 58 - 50) * 340 / 320 };
  const transform = `translate(${169 * (1 - scale) + (target.x - 169) * stored} ${160 * (1 - scale) + (target.y - 160) * stored}) scale(${scale})`;
  return <div className="steering-demo-state">
    <div className="steering-demo-language"><div>{[prior, entry].map((item, i) => <span key={i} style={{ color: item.language === 'en' ? '#111' : '#8B1E1E' }}>{item.label}</span>)}</div></div>
    <svg className="steering-demo-state-map" viewBox="0 10 340 330">
      <defs>
        <clipPath id={`${id}-state`}><rect x="30" y="18" width="280" height="280" /></clipPath>
        <clipPath id={`${id}-shift`}><rect className="steering-demo-shift-mask" x="30" y="18" width={swipe * 280} height="280" /></clipPath>
        <clipPath id={`${id}-base`}><rect className="steering-demo-base-mask" x={30 + swipe * 280} y="18" width={(1 - swipe) * 280} height="280" /></clipPath>
      </defs>
      <g clipPath={`url(#${id}-state)`}>
        <g clipPath={`url(#${id}-base)`}><g className="steering-demo-field-english"><Field language={prior.language} /></g></g>
        <g clipPath={`url(#${id}-shift)`}><g className="steering-demo-field-chinese"><Field language={entry.language} /></g></g>
      </g>
      <g className="steering-demo-route" opacity={phase === 'steering' || phase === 'steered' ? 1 : 0} clipPath={`url(#${id}-state)`}>
        <path className="steering-demo-route-line" d={route?.d} fill="none" stroke="#007C7C" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" pathLength="1" strokeDasharray="1" strokeDashoffset={1 - ease(snapshot.corruption)} />
        <circle className="steering-demo-route-start" cx={route?.start[0]} cy={route?.start[1]} r="5" fill="#8B1E1E" stroke="white" strokeWidth="2" />
        <circle className="steering-demo-route-end" cx={route?.end[0]} cy={route?.end[1]} r="5" fill="#007C7C" stroke="white" strokeWidth="2" />
      </g>
      <g className="steering-demo-axis-labels" fill="#111111"><text x="170" y="327" textAnchor="middle">Dim 1</text><text x="9" y="158" textAnchor="middle" transform="rotate(-90 9 158)">Dim 2</text></g>
      <g className="steering-demo-direction" style={{ opacity: directionOpacity }} transform={transform}>
        <path className="steering-demo-direction-line" d="M84 247 254 72" fill="none" stroke="#007C7C" strokeWidth="3" pathLength="1" style={{ opacity: drawn, strokeDasharray: 1, strokeDashoffset: 1 - drawn }} />
        <path className="steering-demo-direction-head" d="m237 75 17-3-2 17" fill="none" stroke="#007C7C" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: Math.max(0, Math.min(1, drawn * 4 - 3)) }} />
        <circle cx="84" cy="247" r="5" fill="#111111" stroke="white" strokeWidth="2" />
        <circle cx="254" cy="72" r="5" fill="#007C7C" stroke="white" strokeWidth="2" />
      </g>
      <g className="steering-demo-direction-labels" fill="#111111" style={{ opacity: labelsOpacity }}><text x="50" y="272">More English</text><text x="177" y="55">More Spanish</text></g>
    </svg>
  </div>;
}

function Conversation({ snapshot }) {
  return <>
    <div className="steering-demo-scene">
      <div className="steering-demo-question">
        <svg className="steering-demo-asker" viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="3.25"><circle cx="32" cy="20" r="10" /><path d="M12 54v-5c0-12 9-18 20-18s20 6 20 18v5" strokeLinecap="round" /></svg>
        <div className="steering-demo-copy"><span className="steering-demo-label">Question:</span><p className="steering-demo-question-text" lang={snapshot.questionLanguage} data-typing={snapshot.questionTyping}>{snapshot.question}</p></div>
      </div>
      <div className="steering-demo-answer">
        <div className="steering-demo-copy"><span className="steering-demo-label">Model response:</span><p className="steering-demo-answer-text" lang={snapshot.answerLanguage} data-typing={snapshot.answerTyping}>{snapshot.characters.map(({ char, changing }, i) => <span key={i} className={`steering-demo-char is-visible${changing ? ' is-glitch' : ''}`}>{char}</span>)}</p></div>
        <div className="steering-demo-model"><svg className="steering-demo-robot" viewBox="0 0 100 94" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round"><path d="M50 26V16" /><circle cx="50" cy="11" r="5" strokeWidth="3" /><rect x="15" y="29" width="70" height="57" rx="13" /><path d="M15 44H8v23h7M85 44h7v23h-7" /><ellipse cx="35" cy="51" rx="7" ry="9" /><ellipse cx="65" cy="51" rx="7" ry="9" /><path d="M40 70q10 5 20 0" strokeWidth="4" /><path d="M27 10h10M32 5v10M65 18h8M69 14v8" stroke="#999999" strokeWidth="2.5" /></svg></div>
      </div>
    </div>
    <div className="steering-demo-stimulation"><svg className="steering-demo-bolt" viewBox="0 0 60 76"><path d="M36 4 56 15 37 28 45 33 26 45 32 50 6 71 23 43 18 39 31 24 26 20Z" fill="#007C7C" stroke="#111111" strokeWidth="2.5" strokeLinejoin="round" /></svg></div>
  </>;
}

function usePlayback(root, active, elapsedMs, maxFps) {
  const [paused, setPaused] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [elapsed, setElapsed] = useState(0);
  const [visible, setVisible] = useState(false);
  const [foreground, setForeground] = useState(!document.hidden);
  const [replayVersion, setReplayVersion] = useState(0);
  const clock = useRef(0);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { threshold: .25 });
    observer.observe(root.current);
    const visibility = () => setForeground(!document.hidden);
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => { setPaused(motion.matches); clock.current = 0; setElapsed(0); };
    document.addEventListener('visibilitychange', visibility);
    motion.addEventListener('change', change);
    return () => { observer.disconnect(); document.removeEventListener('visibilitychange', visibility); motion.removeEventListener('change', change); };
  }, [root]);
  const running = active && visible && foreground && !paused && elapsedMs === undefined;
  useEffect(() => {
    if (!running) return;
    let frame, previous = null, lastPaint = -Infinity;
    const tick = now => {
      if (previous !== null) clock.current += Math.min(now - previous, 100);
      previous = now;
      if (!maxFps || now - lastPaint >= 1000 / maxFps - .5) {
        setElapsed(clock.current);
        lastPaint = now;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [running, replayVersion, maxFps]);
  return { elapsed: elapsedMs ?? elapsed, paused, running, toggle: () => setPaused(value => !value), replay: () => { clock.current = 0; setElapsed(0); setReplayVersion(value => value + 1); } };
}

export function RobustSteeringLandscape(props) { return <RobustSteeringAnimation {...props} part="landscape" />; }
export function RobustSteeringConversation(props) { return <RobustSteeringAnimation {...props} part="conversation" />; }

export default function RobustSteeringAnimation({ compact = false, active = true, onClose, part = 'full', elapsedMs, maxFps }) {
  const root = useRef(null);
  const playback = usePlayback(root, active, elapsedMs, maxFps);
  const snapshot = sampleSteering(playback.elapsed);
  const view = part === 'landscape' ? { x: 20, y: 12, width: 320, height: compact ? 349 : 397 }
    : part === 'conversation' ? { x: 350, y: 50, width: 416, height: compact ? 270 : 310 }
    : { x: 0, y: 0, width: 776, height: 380 };
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => root.current?.style.setProperty('--steering-scale', Math.min(entry.contentRect.width / view.width, entry.contentRect.height / view.height)));
    observer.observe(root.current);
    return () => observer.disconnect();
  }, [view.width, view.height]);
  return <div className={`rs-animation${compact ? ' rs-compact' : ''}`} ref={root} data-part={part}
    onKeyDown={event => { if (event.key === 'Escape') onClose?.(); }}
    style={{ aspectRatio: `${view.width} / ${view.height}`, '--steering-center-x': `${view.x + view.width / 2}px`, '--steering-center-y': `${view.y + view.height / 2}px` }}>
    <div className="rs-stage steering-demo" data-phase={snapshot.phase} data-language={snapshot.entry.language}
      data-paused={!playback.running} style={{ '--demo-swipe': snapshot.swipe }} aria-hidden="true">
      {part !== 'conversation' && <Landscape snapshot={snapshot} />}
      {part !== 'landscape' && <Conversation snapshot={snapshot} />}
    </div>
    {!compact && <>
      <span className="rs-description" aria-live="polite">{snapshot.phase === 'steered' ? `Steered answer: ${snapshot.entry.steered}` : `${snapshot.entry.label} question: ${snapshot.entry.question} Answer: ${snapshot.entry.baseline}`}</span>
      <div className="rs-controls">
        <button type="button" onClick={playback.toggle} aria-label={playback.paused ? 'Play robust steering animation' : 'Pause robust steering animation'} title={playback.paused ? 'Play' : 'Pause'}>{playback.paused ? <Play size={17} /> : <Pause size={17} />}</button>
        <button type="button" onClick={playback.replay} aria-label="Replay robust steering animation" title="Replay"><RotateCcw size={17} /></button>
      </div>
    </>}
  </div>;
}
