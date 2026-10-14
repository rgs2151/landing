import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowUpRight, ChevronDown, Pause, Play, RotateCcw } from 'lucide-react';
import { profile, paper, robustSteering, neuralFieldManifold } from '../../content/site';
import ProjectVideo from '../../shared/ProjectVideo';
import NeuralFieldAnimation, { NeuralFieldSignal, NeuralFieldAutoregressive, NeuralFieldManifold } from '../../shared/NeuralFieldAnimation';
import RobustSteeringAnimation, { RobustSteeringLandscape, RobustSteeringConversation } from '../../shared/RobustSteeringAnimation';
import './styles.css';
import NaturePreview, { NaturePreviewClock } from './NaturePreviews';

const projects = { cooperation: paper, 'robust-steering': robustSteering, 'neural-field-manifold': neuralFieldManifold };
const names = { cooperation: 'Leader follower', 'robust-steering': robustSteering.shortTitle, 'neural-field-manifold': neuralFieldManifold.title.split(':')[0] };
const publication = id => id === 'cooperation' ? `Nature ${paper.year}`
  : projects[id].venue ? `${projects[id].venue} ${projects[id].year} · ${projects[id].status}`
  : `${projects[id].status} · ${projects[id].year}`;
const selectedFromHash = () => {
  const id = location.hash.slice(1);
  return projects[id] || id === 'about' ? id : null;
};
const pieces = [
  { id: 'nature-0', project: 'cooperation', clip: 0, x: 30, y: 148, width: 180, angle: -8, fan: [-18,-12] },
  { id: 'nature-1', project: 'cooperation', clip: 1, x: 690, y: 4, width: 200, angle: 8, fan: [10,-10] },
  { id: 'nature-2', project: 'cooperation', clip: 2, x: 260, y: 490, width: 200, angle: 6, fan: [-10,14] },
  { id: 'nature-3', project: 'cooperation', clip: 3, x: 970, y: 410, width: 170, angle: -9, fan: [18,8] },
  { id: 'landscape', project: 'robust-steering', Component: RobustSteeringLandscape, x: 940, y: 130, width: 250, angle: 5, fan: [6,-12] },
  { id: 'conversation', project: 'robust-steering', Component: RobustSteeringConversation, x: 490, y: 504, width: 350, angle: -3, fan: [8,12] },
  { id: 'signal', project: 'neural-field-manifold', Component: NeuralFieldSignal, x: 290, y: 15, width: 310, angle: -5, fan: [-12,-10] },
  { id: 'autoregressive', project: 'neural-field-manifold', Component: NeuralFieldAutoregressive, x: 870, y: 560, width: 270, angle: 5, fan: [12,8] },
  { id: 'manifold', project: 'neural-field-manifold', Component: NeuralFieldManifold, x: 20, y: 370, width: 190, angle: -6, fan: [-12,8] },
];

function Navigation({ selected, open, hover }) {
  const [menu, setMenu] = useState(null);
  const root = useRef(null);
  useEffect(() => {
    const dismiss = event => { if (!root.current?.contains(event.target)) setMenu(null); };
    document.addEventListener('pointerdown', dismiss);
    return () => document.removeEventListener('pointerdown', dismiss);
  }, []);
  useEffect(() => setMenu(null), [selected]);
  return <nav className="pb-navigation" aria-label="Main navigation" ref={root}
    onPointerLeave={() => { setMenu(null); hover(null); }}
    onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) { setMenu(null); hover(null); } }}
    onKeyDown={event => { if (event.key === 'Escape' && menu) { event.stopPropagation(); setMenu(null); root.current.querySelector(`[data-menu="${menu}"]`)?.focus(); } }}>
    <div className="pb-nav-group" onPointerEnter={() => setMenu('research')}>
      <button data-menu="research" aria-expanded={menu === 'research'} aria-controls="pb-research-menu"
        onClick={() => setMenu('research')}
        onKeyDown={event => { if (event.key === 'ArrowDown') { event.preventDefault(); setMenu('research'); requestAnimationFrame(() => root.current.querySelector('.pb-dropdown a')?.focus()); } }}>Research<ChevronDown size={14} /></button>
      <div id="pb-research-menu" className="pb-dropdown" hidden={menu !== 'research'}>
        {Object.keys(projects).map(id => <a key={id} href={`#${id}`} aria-current={selected === id ? 'page' : undefined}
          onClick={event => { event.preventDefault(); setMenu(null); open(id, event.currentTarget); }}
          onPointerEnter={() => hover(id)} onFocus={() => hover(id)}>
          <span>{projects[id].title}</span><small>{publication(id)}</small>
        </a>)}
      </div>
    </div>
    <div className="pb-nav-group" onPointerEnter={() => setMenu('packages')}>
      <button data-menu="packages" aria-expanded={menu === 'packages'} aria-controls="pb-packages-menu" onClick={() => setMenu('packages')}>Tool Packages<ChevronDown size={14} /></button>
      <div id="pb-packages-menu" className="pb-dropdown pb-packages" hidden={menu !== 'packages'}><p>No packages listed yet.</p></div>
    </div>
    <a href="#about" aria-current={selected === 'about' ? 'page' : undefined} onPointerEnter={() => setMenu(null)}
      onClick={event => { event.preventDefault(); open('about', event.currentTarget); }}>About</a>
  </nav>;
}

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
    <p className="pb-authors">{content.authors.map((author, index) => <span key={author}>{index > 0 ? ', ' : ''}<span className={author === profile.name ? 'pb-self' : undefined}>{author}</span>{content.equalContribution?.includes(author) && <sup>*</sup>}</span>)}</p>
    {content.equalContribution && <p className="pb-equal">* Equal contribution</p>}
    <div className="pb-project-links">
      <a href={content.url} target="_blank" rel="noopener noreferrer">{id === 'cooperation' ? <><img src={paper.natureLogo} alt="Nature" /> {content.year}</> : publication(id)}<ArrowUpRight size={17} /></a>
      <a href={content.code} target="_blank" rel="noopener noreferrer"><img src={paper.githubLogo} alt="GitHub" /> code<ArrowUpRight size={17} /></a>
    </div>
    <div className="pb-project-media">
      {id === 'cooperation' && <NatureVideo />}
      {id === 'robust-steering' && <RobustSteeringAnimation />}
      {id === 'neural-field-manifold' && <NeuralFieldAnimation />}
    </div>
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
  const [viewport, setViewport] = useState(() => ({ width: innerWidth, height: innerHeight }));
  const scene = useRef(null);
  const heading = useRef(null);
  const opener = useRef(null);
  const active = selected || hovered;
  const stackX = (24 - (viewport.width - 1200 * scale) / 2) / scale;
  const stackY = (viewport.height - 220 - 92) / scale;
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
    const resize = () => {
      setScale(Math.min(scene.current.clientWidth / 1240, (innerHeight - 142) / 760, 1.12));
      setViewport({ width: innerWidth, height: innerHeight });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(scene.current);
    window.addEventListener('resize', resize);
    return () => { observer.disconnect(); window.removeEventListener('resize', resize); };
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
      const target = opener.current?.isConnected && opener.current.getClientRects().length ? opener.current : document.querySelector('.pb-title');
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
    <header className="pb-header" data-open={Boolean(selected)}>
      {selected && <div className="pb-identity">
        <button className="pb-portrait" onClick={event => open('about', event.currentTarget)} aria-label={`About ${profile.name}`} title={`About ${profile.name}`}><img src={profile.photo} alt={profile.name} /></button>
        <div><h1><button onClick={event => open('about', event.currentTarget)}>{profile.name}</button></h1><span>{profile.email}</span><a href={profile.github} target="_blank" rel="noopener noreferrer">{profile.githubLabel}<ArrowUpRight size={13} /></a></div>
      </div>}
      <Navigation selected={selected} open={open} hover={setHovered} />
    </header>

    <section className="pb-stage" ref={scene} data-open={Boolean(selected)} aria-label="Interactive research composition" onPointerLeave={() => setHovered(null)}>
      <div className="pb-canvas" style={{ '--pb-scale': scale }} data-open={Boolean(selected)} data-active={active || ''} data-paused={paused || Boolean(selected)}>
        <div className="pb-lettering" aria-hidden={Boolean(selected)} inert={selected ? true : undefined}>
          <h1><button className="pb-title" onClick={event => open('about',event.currentTarget)} aria-label={`About ${profile.name}`}><span>{profile.name.split(' ')[0]}</span><span>{profile.name.split(' ').slice(1).join(' ')}</span></button></h1>
        </div>
        <NaturePreviewClock active={!paused && !selected} replay={replay}>{pieces.map((piece,index) => {
          const matching = active === piece.project;
          const factor = selected ? Math.min(110 / piece.width, .5) : 1;
          const x = selected ? stackX + 30 + index % 3 * 24 : piece.x + (matching ? piece.fan[0] : 0);
          const y = selected ? stackY + 10 + index % 4 * 15 : piece.y + (matching ? piece.fan[1] : 0);
          return <button key={piece.id} className="pb-fragment" data-project={piece.project} data-matching={matching}
            style={{ width: piece.width, transform: `translate(${x}px, ${y}px) rotate(${selected ? (index - 4) * 5 : matching ? 0 : piece.angle}deg) scale(${factor})`, '--pb-delay': `${-index * .8}s` }}
            aria-label={`${names[piece.project]}: ${piece.clip === undefined ? piece.id : `video ${piece.clip + 1}`}`}
            tabIndex={selected ? -1 : 0} aria-hidden={selected ? true : undefined} aria-controls="pb-reading"
            onClick={event => selected ? close() : open(piece.project,event.currentTarget)} onPointerEnter={() => { if (!selected) setHovered(piece.project); }} onFocus={() => setHovered(piece.project)}>
            <div className="pb-fragment-art" key={replay}>
              {piece.Component ? <piece.Component compact active={!paused && !selected} /> : <NaturePreview clip={piece.clip} poster={paper.clips[piece.clip].poster} />}
            </div>
          </button>;
        })}</NaturePreviewClock>
        {selected && <button className="pb-stack-return" style={{ transform: `translate(${stackX}px, ${stackY}px)` }} onClick={close} aria-label="Return to composition" title="Return to composition"><ArrowLeft size={22} /></button>}
      </div>
      {selected && <section id="pb-reading" className="pb-reading" key={selected}>
        <ProjectDetail id={selected} headingRef={heading} />
      </section>}
    </section>
    <footer className="pb-footer">
      {!selected && !reduced && <div className="pb-scene-controls">
        <button onClick={() => setPaused(value => !value)} title={paused ? 'Play' : 'Pause'} aria-label={paused ? 'Play composition' : 'Pause composition'}>{paused ? <Play size={17} /> : <Pause size={17} />}</button>
        <button onClick={() => setReplay(value => value + 1)} title="Replay" aria-label="Replay composition"><RotateCcw size={17} /></button>
      </div>}
      <span>&copy; 2026 {profile.name}</span>
    </footer>
  </main>;
}
