import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowUpRight, Pause, Play, RotateCcw } from 'lucide-react';
import { profile, paper, robustSteering, neuralFieldManifold } from '../../content/site';
import ProjectVideo from '../../shared/ProjectVideo';
import NeuralFieldAnimation, { NeuralFieldSignal, NeuralFieldAutoregressive, NeuralFieldManifold } from '../../shared/NeuralFieldAnimation';
import RobustSteeringAnimation, { RobustSteeringLandscape, RobustSteeringConversation } from '../../shared/RobustSteeringAnimation';
import './styles.css';

const projects = { cooperation: paper, 'robust-steering': robustSteering, 'neural-field-manifold': neuralFieldManifold };
const names = { cooperation: `Nature ${paper.year}`, 'robust-steering': robustSteering.shortTitle, 'neural-field-manifold': neuralFieldManifold.title.split(':')[0] };
const selectedFromHash = () => {
  const id = location.hash.slice(1);
  return projects[id] || id === 'about' ? id : null;
};
const pieces = [
  { id: 'nature-0', project: 'cooperation', clip: 0, x: 30, y: 200, width: 210, angle: -7, fan: [-14,-10] },
  { id: 'nature-1', project: 'cooperation', clip: 1, x: 335, y: 8, width: 190, angle: 5, fan: [-8,-8] },
  { id: 'nature-2', project: 'cooperation', clip: 2, x: 380, y: 482, width: 180, angle: -5, fan: [8,12] },
  { id: 'nature-3', project: 'cooperation', clip: 3, x: 65, y: 480, width: 150, angle: 4, fan: [-12,8] },
  { id: 'landscape', project: 'robust-steering', Component: RobustSteeringLandscape, x: 810, y: 140, width: 220, angle: 4, fan: [12,-10] },
  { id: 'conversation', project: 'robust-steering', Component: RobustSteeringConversation, x: 880, y: 475, width: 270, angle: -4, fan: [12,8] },
  { id: 'signal', project: 'neural-field-manifold', Component: NeuralFieldSignal, x: 565, y: 18, width: 215, angle: -4, fan: [-8,-8] },
  { id: 'autoregressive', project: 'neural-field-manifold', Component: NeuralFieldAutoregressive, x: 630, y: 480, width: 220, angle: 3, fan: [-10,10] },
  { id: 'manifold', project: 'neural-field-manifold', Component: NeuralFieldManifold, x: 1045, y: 290, width: 145, angle: 5, fan: [10,-6] },
];

function NatureVideo() {
  const host = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [enabled, setEnabled] = useState(() => !matchMedia('(prefers-reduced-motion: reduce)').matches);
  function toggle() {
    const video = host.current.querySelector('video');
    if (!video.paused) { setEnabled(false); video.pause(); }
    else { setEnabled(true); video.play().catch(() => setPlaying(false)); }
  }
  return <div className="pb-video" ref={host}>
    <ProjectVideo src={paper.strip} poster={paper.stripPoster} compact active={enabled}
      aria-label={paper.title} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
    <div className="pb-media-controls">
      <button type="button" onClick={toggle} title={playing ? 'Pause' : 'Play'} aria-label={playing ? 'Pause Nature video' : 'Play Nature video'}>{playing ? <Pause size={17} /> : <Play size={17} />}</button>
      <button type="button" onClick={() => { host.current.querySelector('video').currentTime = 0; }} title="Replay" aria-label="Replay Nature video"><RotateCcw size={17} /></button>
    </div>
  </div>;
}

function ProjectDetail({ id, headingRef }) {
  if (id === 'about') return <article className="pb-about">
    <h2 tabIndex={-1} ref={headingRef}>{profile.name}</h2>
    <img src={profile.photo} alt={profile.name} />
    <p>{profile.role} <a href={profile.institutionUrl}>{profile.institution}</a>{profile.affiliation} {profile.statement}</p>
    <p>{profile.email}<br /><a href={profile.github}>{profile.githubLabel}</a></p>
  </article>;
  const content = projects[id];
  return <article className="pb-project" aria-labelledby="pb-project-heading">
    <h2 id="pb-project-heading" tabIndex={-1} ref={headingRef}>{content.title}</h2>
    <div className="pb-project-media">
      {id === 'cooperation' && <NatureVideo />}
      {id === 'robust-steering' && <RobustSteeringAnimation />}
      {id === 'neural-field-manifold' && <NeuralFieldAnimation />}
    </div>
    <div className="pb-project-links">
      <a href={content.url} target="_blank" rel="noopener noreferrer">{id === 'cooperation' ? <img src={paper.natureLogo} alt="Nature" /> : 'Project'} {content.year}<ArrowUpRight size={17} /></a>
      <a href={content.code} target="_blank" rel="noopener noreferrer"><img src={paper.githubLogo} alt="GitHub" /> code<ArrowUpRight size={17} /></a>
    </div>
    <p className="pb-authors">{content.authors.map((author, index) => <span key={author}>{index > 0 ? ', ' : ''}<span className={author === profile.name ? 'pb-self' : undefined}>{author}</span>{content.equalContribution?.includes(author) && <sup>*</sup>}</span>)}</p>
    {content.equalContribution && <p className="pb-equal">* Equal contribution</p>}
    <div className="pb-blurb">{content.blurb.split(/\n\s*\n/).map((paragraph,index) => <p key={index}>{paragraph}</p>)}</div>
  </article>;
}

export default function PrototypeB() {
  const [desktop, setDesktop] = useState(() => matchMedia('(min-width: 901px)').matches);
  const [selected, setSelected] = useState(selectedFromHash);
  const [hovered, setHovered] = useState(null);
  const [paused, setPaused] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [reduced, setReduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [replay, setReplay] = useState(0);
  const [scale, setScale] = useState(1);
  const scene = useRef(null);
  const heading = useRef(null);
  const opener = useRef(null);
  const active = selected || hovered;
  useEffect(() => {
    const media = matchMedia('(min-width: 901px)');
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const motionChange = () => { setReduced(motion.matches); setPaused(motion.matches); };
    const change = () => setDesktop(media.matches);
    const hash = () => { setSelected(selectedFromHash()); setHovered(null); };
    media.addEventListener('change', change); window.addEventListener('hashchange', hash);
    motion.addEventListener('change', motionChange);
    return () => { media.removeEventListener('change', change); motion.removeEventListener('change', motionChange); window.removeEventListener('hashchange', hash); };
  }, []);
  useEffect(() => {
    if (!desktop || !scene.current) return;
    const observer = new ResizeObserver(([entry]) => setScale(Math.min(entry.contentRect.width / 1240, 1.12)));
    observer.observe(scene.current);
    return () => observer.disconnect();
  }, [desktop]);
  useEffect(() => {
    if (!selected || !desktop) return;
    heading.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [selected, desktop]);
  function open(id, target) {
    opener.current = target;
    location.hash = id;
    setSelected(id); setHovered(null);
  }
  function close() {
    history.pushState(null, '', location.pathname + location.search);
    setSelected(null); setHovered(null);
    requestAnimationFrame(() => {
      const target = opener.current?.isConnected ? opener.current : document.querySelector('.pb-navigation a');
      target?.focus({ preventScroll: true });
    });
  }
  useEffect(() => {
    const pop = () => setSelected(selectedFromHash());
    const escape = event => { if (event.key === 'Escape' && selected) close(); };
    window.addEventListener('popstate', pop); window.addEventListener('keydown', escape);
    return () => { window.removeEventListener('popstate', pop); window.removeEventListener('keydown', escape); };
  }, [selected]);

  if (!desktop) return <main className="prototype-b pb-mobile">
    <img src={profile.photo} alt={profile.name} /><h1>{profile.name}</h1><p>{profile.statement}</p>
    <p>{profile.email}<br /><a href={profile.github}>{profile.githubLabel}</a></p>
    <p>Open on a computer for the full experience.</p>
  </main>;

  return <main className="prototype-b" data-selected={selected || ''}>
    <header className="pb-header">
      <div className="pb-identity">
        <button className="pb-portrait" onClick={event => open('about', event.currentTarget)} aria-label={`About ${profile.name}`} title={`About ${profile.name}`}><img src={profile.photo} alt={profile.name} /></button>
        <div><h1>{profile.name}</h1><span>{profile.email}</span><a href={profile.github} target="_blank" rel="noopener noreferrer">{profile.githubLabel}<ArrowUpRight size={13} /></a></div>
      </div>
      <nav className="pb-navigation" aria-label="Projects">
        {Object.keys(projects).map(id => <a key={id} href={`#${id}`} aria-current={selected === id ? 'page' : undefined}
          onClick={event => { event.preventDefault(); open(id, event.currentTarget); }}
          onPointerEnter={() => setHovered(id)} onPointerLeave={() => setHovered(null)} onFocus={() => setHovered(id)} onBlur={() => setHovered(null)}>{names[id]}</a>)}
      </nav>
    </header>

    <section className="pb-stage" ref={scene} data-open={Boolean(selected)} aria-label="Interactive research composition" onPointerLeave={() => setHovered(null)}>
      <div className="pb-canvas" style={{ '--pb-scale': scale }} data-open={Boolean(selected)} data-active={active || ''} data-paused={paused || Boolean(selected)}>
        <div className="pb-lettering" aria-hidden={Boolean(selected)} inert={selected ? true : undefined}>
          <button className="pb-title pb-title-nature" onClick={event => open('cooperation',event.currentTarget)} onPointerEnter={() => setHovered('cooperation')} onFocus={() => setHovered('cooperation')} aria-label={paper.title}><span>leader</span><span>follower</span><ArrowUpRight size={38} /></button>
          <button className="pb-title pb-title-steering" onClick={event => open('robust-steering',event.currentTarget)} onPointerEnter={() => setHovered('robust-steering')} onFocus={() => setHovered('robust-steering')}>{robustSteering.shortTitle}</button>
          <button className="pb-title pb-title-manifold" onClick={event => open('neural-field-manifold',event.currentTarget)} onPointerEnter={() => setHovered('neural-field-manifold')} onFocus={() => setHovered('neural-field-manifold')}>{names['neural-field-manifold']}</button>
          <a className="pb-nature-mark" href={paper.url} target="_blank" rel="noopener noreferrer"><img src={paper.natureLogo} alt="Nature" /> {paper.year}<ArrowUpRight size={18} /></a>
        </div>
        {pieces.map((piece,index) => {
          const matching = active === piece.project;
          const factor = selected ? Math.min(140 / piece.width, .65) : 1;
          const x = selected ? 18 + index % 2 * 12 : piece.x + (matching ? piece.fan[0] : 0);
          const y = selected ? 18 + index * 84 : piece.y + (matching ? piece.fan[1] : 0);
          return <button key={piece.id} className="pb-fragment" data-project={piece.project} data-matching={matching}
            style={{ width: piece.width, transform: `translate(${x}px, ${y}px) rotate(${selected || matching ? 0 : piece.angle}deg) scale(${factor})`, '--pb-delay': `${-index * .8}s` }}
            aria-label={`${names[piece.project]}: ${piece.clip === undefined ? piece.id : `video ${piece.clip + 1}`}`}
            aria-expanded={selected === piece.project} aria-controls="pb-reading"
            onClick={event => open(piece.project,event.currentTarget)} onPointerEnter={() => setHovered(piece.project)} onFocus={() => setHovered(piece.project)}>
            <div className="pb-fragment-art" key={replay}>
              {piece.Component ? <piece.Component compact active={!paused && !selected} /> : <ProjectVideo {...paper.clips[piece.clip]} compact active={!paused && !selected} aria-label={`${paper.title}, video ${piece.clip + 1}`} />}
            </div>
          </button>;
        })}
      </div>
      {selected && <section id="pb-reading" className="pb-reading" key={selected}>
        <button className="pb-return" onClick={close} aria-label="Return to composition" title="Return to composition"><ArrowLeft size={22} /></button>
        <ProjectDetail id={selected} headingRef={heading} />
      </section>}
    </section>
    <footer className="pb-footer">
      <button onClick={event => open('about',event.currentTarget)}>About</button>
      {!selected && !reduced && <div className="pb-scene-controls">
        <button onClick={() => setPaused(value => !value)} title={paused ? 'Play' : 'Pause'} aria-label={paused ? 'Play composition' : 'Pause composition'}>{paused ? <Play size={17} /> : <Pause size={17} />}</button>
        <button onClick={() => setReplay(value => value + 1)} title="Replay" aria-label="Replay composition"><RotateCcw size={17} /></button>
      </div>}
      <span>&copy; 2026 {profile.name}</span>
    </footer>
  </main>;
}
