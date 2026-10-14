import { useEffect, useId, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import * as THREE from 'three';
import { loadNeuralFieldModel, advanceFrame, traceRanges, PLAYBACK_SPEED, SIGNAL_WINDOW, SIGNAL_HEIGHT, waveformPath } from './neuralFieldModel';
import './neural-field-animation.css';

const ACCENT = '#8b1e1e';
const VIEWBOXES = {
  full: '0 50 1000 480',
  signal: '8 220 250 165',
  autoregressive: '312 375 375 185',
  manifold: '740 190 270 255',
};

export function NeuralFieldSignal(props) {
  return <NeuralFieldAnimation {...props} part="signal" />;
}

export function NeuralFieldAutoregressive(props) {
  return <NeuralFieldAnimation {...props} part="autoregressive" />;
}

export function NeuralFieldManifold(props) {
  return <NeuralFieldAnimation {...props} part="manifold" />;
}

function createManifold(host, model) {
  const LAG = model.lag;
  const camera = new THREE.PerspectiveCamera(33, 1, 0.1, 30);
  camera.position.set(3.2, 3.1, 4.9).multiplyScalar(1.2);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  } catch {
    // Keep the scientific view usable when WebGL is disabled.
    const canvas = document.createElement('canvas');
    host.append(canvas);
    const context = canvas.getContext('2d');
    const projected = new THREE.Vector3();
    return {
      resize(width, height) {
        canvas.width = width * 2; canvas.height = height * 2;
        camera.aspect = width / height; camera.updateProjectionMatrix();
      },
      draw(frame) {
        const { width, height } = canvas;
        context.clearRect(0, 0, width, height);
        context.strokeStyle = ACCENT;
        context.lineWidth = 1.6;
        context.beginPath();
        for (const { start, count } of traceRanges(frame, model.signal.length, LAG)) {
          for (let i = start; i < start + count; i++) {
            projected.fromArray(model.points, i * 3).project(camera);
            const px = (projected.x + 1) * width / 2;
            const py = (1 - projected.y) * height / 2;
            if (i === start) context.moveTo(px, py);
            else context.lineTo(px, py);
          }
        }
        context.stroke();
      },
      dispose() { canvas.remove(); },
    };
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xffffff, 0);
  host.append(renderer.domElement);
  const scene = new THREE.Scene();
  const position = new THREE.BufferAttribute(model.points, 3);
  const geometries = [new THREE.BufferGeometry(), new THREE.BufferGeometry()];
  const material = new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.78 });
  geometries.forEach(geometry => {
    geometry.setAttribute('position', position);
    scene.add(new THREE.Line(geometry, material));
  });
  const pointGeometry = new THREE.SphereGeometry(0.065, 12, 8);
  const pointMaterial = new THREE.MeshBasicMaterial({ color: 0x191919 });
  const point = new THREE.Mesh(pointGeometry, pointMaterial);
  scene.add(point);
  const grid = new THREE.GridHelper(4.4, 8, 0xcccccc, 0xe0e0e0);
  grid.position.y = -1.85;
  scene.add(grid);
  return {
    resize(width, height) {
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    },
    draw(frame) {
      const ranges = traceRanges(frame, model.signal.length, LAG);
      geometries.forEach((geometry, index) => {
        const range = ranges[index];
        geometry.setDrawRange(range?.start || 0, range?.count || 0);
      });
      point.visible = frame >= 2 * LAG;
      point.position.fromArray(model.points, frame * 3);
      renderer.render(scene, camera);
    },
    dispose() {
      geometries.forEach(geometry => geometry.dispose());
      material.dispose(); pointGeometry.dispose(); pointMaterial.dispose();
      grid.geometry.dispose(); grid.material.dispose(); renderer.dispose();
      renderer.forceContextLoss(); renderer.domElement.remove();
    },
  };
}

export default function NeuralFieldAnimation(props) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let mounted = true;
    loadNeuralFieldModel().then(value => { if (mounted) setModel(value); })
      .catch(() => { if (mounted) setError(true); });
    return () => { mounted = false; };
  }, [attempt]);
  if (!model) return <div className="nf-animation nf-diagram" data-part={props.part || 'full'} aria-label={error ? 'LFP recording unavailable' : 'Loading LFP recording'}>
    {error && !props.compact && <div className="nf-controls"><button title="Retry loading LFP recording" aria-label="Retry loading LFP recording"
      onClick={() => { setError(false); setAttempt(value => value + 1); }}><RotateCcw size={17} /></button></div>}
  </div>;
  return <RecordedNeuralFieldAnimation {...props} model={model} />;
}

function RecordedNeuralFieldAnimation({ compact = false, active = true, part = 'full', model }) {
  const full = part === 'full';
  const LAG = model.lag;
  const id = useId().replaceAll(':', '');
  const root = useRef(null);
  const host = useRef(null);
  const waveform = useRef(null);
  const lagPoints = useRef([]);
  const temporalPoint = useRef(null);
  const controller = useRef(null);
  const [paused, setPaused] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);

  useEffect(() => {
    const manifold = host.current ? createManifold(host.current, model) : null;
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const initialFrame = Math.floor(model.signal.length / 2);
    let frame = initialFrame;
    let raf = 0;
    let last = 0;
    let remainder = 0;
    let visible = false;
    let playing = true;
    let motionAllowed = !reducedMotion.matches;
    function draw() {
      waveform.current?.setAttribute('d', waveformPath(model.signal, frame));
      lagPoints.current.forEach((point, j) => {
        if (!point) return;
        const index = (frame - j * LAG + model.signal.length) % model.signal.length;
        point.setAttribute('cx', 242 - j * LAG * 224 / (SIGNAL_WINDOW - 1));
        point.setAttribute('cy', 286 - model.signal[index] * SIGNAL_HEIGHT);
      });
      if (temporalPoint.current) temporalPoint.current.style.visibility = frame >= 2 * LAG ? 'visible' : 'hidden';
      if (temporalPoint.current && frame >= 2 * LAG) {
        temporalPoint.current.setAttribute('cx', 567 + model.signal[frame - LAG] * 19);
        temporalPoint.current.setAttribute('cy', 446 - model.signal[frame - 2 * LAG] * 14);
      }
      root.current.dataset.frame = String(frame);
      manifold?.draw(frame);
    }
    function tick(now) {
      if (last) remainder += Math.min(now - last, 100) * model.sampleRate * PLAYBACK_SPEED / 1000;
      last = now;
      const steps = Math.floor(remainder);
      if (steps) {
        remainder -= steps;
        frame = advanceFrame(frame, steps, model.signal.length, 2 * LAG);
        draw();
      }
      raf = requestAnimationFrame(tick);
    }
    function sync() {
      cancelAnimationFrame(raf);
      last = 0;
      if (visible && playing && motionAllowed && !document.hidden) raf = requestAnimationFrame(tick);
    }
    controller.current = {
      setPlaying(value) { playing = value; sync(); },
      replay() { frame = initialFrame; remainder = 0; draw(); },
      allowMotion() { motionAllowed = true; sync(); },
    };
    const resize = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width && height && manifold) { manifold.resize(width, height); draw(); }
    });
    if (host.current) resize.observe(host.current);
    const intersection = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); });
    intersection.observe(root.current);
    const onMotion = () => { motionAllowed = !reducedMotion.matches; sync(); };
    reducedMotion.addEventListener('change', onMotion);
    document.addEventListener('visibilitychange', sync);
    draw();
    return () => {
      cancelAnimationFrame(raf); resize.disconnect(); intersection.disconnect();
      reducedMotion.removeEventListener('change', onMotion);
      document.removeEventListener('visibilitychange', sync);
      manifold?.dispose(); controller.current = null;
    };
  }, [model, LAG, part]);

  useEffect(() => { controller.current?.setPlaying(active && !paused); }, [active, paused]);

  return <div role="figure" className={`nf-animation${compact ? ' nf-compact' : ''}`} data-part={part} ref={root}>
    <div className="nf-diagram">
      <svg viewBox={VIEWBOXES[part]} role="img" aria-labelledby={`${id}-title ${id}-desc`}>
        <title id={`${id}-title`}>Recorded mouse LFP and lag embedding</title>
        <desc id={`${id}-desc`}>Five seconds of recorded mouse hippocampal LFP, using the tutorial's processed samples at 500 Hz and a lag of 10 samples. The upper path shows the hidden dynamical process. The lower path reconstructs state from time delays; the trajectory plots the signal and its two lagged observations directly.</desc>
        <defs>
          <marker id={`${id}-arrow`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L6,3 L0,6" fill="context-stroke" /></marker>
        </defs>
        {full && <text x="18" y="185" className="nf-label">
          Local field<tspan x="18" dy="24">potential<tspan dx="6" fontSize="24" fontFamily="'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', sans-serif">🐭</tspan><tspan dx="4" fontSize="24" fontFamily="'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', sans-serif">🐵</tspan></tspan>
        </text>}
        {(full || part === 'signal') && <>
        <path ref={waveform} fill="none" stroke="#292929" strokeWidth="1.7" />
        {[0, 1, 2, 3].map(j => <circle key={j} ref={el => { lagPoints.current[j] = el; }} r={j ? 3 : 4} fill={ACCENT} opacity={1 - j * 0.15} />)}
        </>}
        {full && <>
        <path d="M186,221 L291,160" className="nf-arrow" markerEnd={`url(#${id}-arrow)`} />
        <text x="205" y="126" textAnchor="middle" className="nf-label">Hidden<tspan x="205" dy="24">dynamic</tspan></text>
        <path d="M186,351 L291,421" className="nf-arrow nf-red" markerEnd={`url(#${id}-arrow)`} />
        <text x="205" y="445" textAnchor="middle" className="nf-label nf-red">Lag<tspan x="205" dy="24">embedding</tspan></text>

        <rect x="306" y="65" width="392" height="190" fill="none" stroke="#898989" strokeWidth="2" />
        <text x="502" y="94" textAnchor="middle" className="nf-label">Hidden dynamical process</text>
        <text x="337" y="157" className="nf-equation">dx</text>
        <path d="M328,165 H354" stroke="currentColor" />
        <text x="337" y="189" className="nf-equation">dt</text>
        <text x="364" y="177" className="nf-equation">= F(x(t); θ)</text>
        <g className="nf-axes" transform="translate(568 177)" markerEnd={`url(#${id}-arrow)`}>
          <path d="M0,0 V-57" /><path d="M0,0 H70" /><path d="M0,0 L-25,39" /><path d="M0,0 L44,36" />
        </g>
        <text x="540" y="128" className="nf-axis-text">x<tspan baselineShift="sub" fontSize="12">n</tspan></text>
        <text x="650" y="176" className="nf-axis-text">x<tspan baselineShift="sub" fontSize="12">3</tspan></text>
        <text x="617" y="232" className="nf-axis-text">x<tspan baselineShift="sub" fontSize="12">2</tspan></text>
        <text x="523" y="226" className="nf-axis-text">x<tspan baselineShift="sub" fontSize="12">1</tspan></text>
        <text x="507" y="198" transform="rotate(-90 507 198)" className="nf-axis-text">State space</text>
        <path d="M594,130 Q613,131 620,157" fill="none" stroke="#555" strokeWidth="3" strokeDasharray="2 10" />

        <rect x="306" y="326" width="392" height="190" fill="#8b1e1e" fillOpacity=".045" stroke={ACCENT} strokeWidth="2.5" />
        <text x="502" y="355" textAnchor="middle" className="nf-label">Autoregressive notion</text>
        </>}
        {(full || part === 'autoregressive') && <>
        <text x="367" y="408" className="nf-equation">x(t) =</text>
        <text x="326" y="441" className="nf-equation" fontSize="32">Σ</text>
        <text x="341" y="420" fontSize="13">p</text>
        <text x="322" y="456" fontSize="12">i = 1</text>
        <text x="353" y="439" className="nf-equation">φ<tspan baselineShift="sub" fontSize="13">i</tspan> x(t−i) +</text>
        <text x="377" y="475" className="nf-equation">ε(t)</text>
        <g className="nf-axes nf-red" transform="translate(569 449)" markerEnd={`url(#${id}-arrow)`}>
          <path d="M0,0 V-58" /><path d="M0,0 H71" /><path d="M0,0 L-26,36" /><path d="M0,0 L45,33" />
        </g>
        <text x="551" y="424" transform="rotate(-90 551 424)" className="nf-axis-text">t − nτ</text>
        <text x="649" y="446" className="nf-axis-text">t − 3τ</text>
        <text x="590" y="504" className="nf-axis-text">t − 2τ</text>
        <text x="546" y="507" transform="rotate(-45 546 507)" className="nf-axis-text">t − τ</text>
        <text x="499" y="478" transform="rotate(-90 499 478)" className="nf-axis-text">Temporal<tspan x="499" dy="18">state space</tspan></text>
        <path d="M594,401 Q615,405 621,429" fill="none" stroke={ACCENT} strokeWidth="3" strokeDasharray="2 10" />
        <circle ref={temporalPoint} r="4" fill={ACCENT} />
        </>}
        {full && <>
        <path d="M713,426 L791,342" className="nf-arrow nf-red" markerEnd={`url(#${id}-arrow)`} />
        <text x="874" y="161" textAnchor="middle" className="nf-label">Formulation of manifold<tspan x="874" dy="24">properties</tspan></text>
        </>}
        {(full || part === 'manifold') && <text x="874" y="392" textAnchor="middle" className="nf-caption">x(t), x(t − τ), x(t − 2τ)</text>}
      </svg>
      {(full || part === 'manifold') && <div className="nf-manifold" ref={host} aria-hidden="true" />}
    </div>
    {!compact && <div className="nf-controls">
      <button type="button" onClick={() => { controller.current?.allowMotion(); setPaused(value => !value); }} aria-label={paused ? 'Play neural field animation' : 'Pause neural field animation'} title={paused ? 'Play' : 'Pause'}>{paused ? <Play size={17} /> : <Pause size={17} />}</button>
      <button type="button" onClick={() => controller.current?.replay()} aria-label="Replay neural field animation" title="Replay"><RotateCcw size={17} /></button>
    </div>}
  </div>;
}
