import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowUpRight, ChevronDown, Pause, Play, RotateCcw } from 'lucide-react';
import { profile, paper, robustSteering, neuralFieldManifold } from '../../content/site';
import ProjectVideo from '../../shared/ProjectVideo';
import ToolDetail from '../../shared/ToolDetail';
import NeuralFieldAnimation, { NeuralFieldSignal, NeuralFieldAutoregressive, NeuralFieldManifold } from '../../shared/NeuralFieldAnimation';
import RobustSteeringAnimation, { RobustSteeringLandscape, RobustSteeringConversation } from '../../shared/RobustSteeringAnimation';
import './styles.css';
import NaturePreview, { NaturePreviewClock } from './NaturePreviews';
import { FreezePreview, TravelingPortrait, useOrbits } from './SceneMotion';
import ProjectTitle from '../../shared/ProjectTitle';
import useProjectArrival from '../../shared/useProjectArrival';

const projects = { cooperation: paper, 'robust-steering': robustSteering, 'neural-field-manifold': neuralFieldManifold };
const names = { cooperation: 'Leader follower', 'robust-steering': robustSteering.shortTitle, 'neural-field-manifold': neuralFieldManifold.title.split(':')[0] };
const publication = (id, omitAccepted = false) => id === 'cooperation' ? `Nature ${paper.year}`
  : omitAccepted && projects[id].venue && projects[id].status === 'Accepted' ? `${projects[id].venue} ${projects[id].year}`
  : projects[id].venue ? `${projects[id].venue} ${projects[id].year} · ${projects[id].status}`
  : `${projects[id].status} · ${projects[id].year}`;
const NO_TOOLS = [];
const DEFAULT_NAVIGATION = { research: 'Research', packages: 'Tool Packages' };
const selectedFromHash = (tools = NO_TOOLS) => {
  const id = location.hash.slice(1);
  return projects[id] || tools.some(tool => tool.id === id) || id === 'about' ? id : null;
};
export const pieces = [
  { id: 'nature-0', project: 'cooperation', clip: 0, x: 205, y: -15, width: 270, angle: -8 },
  { id: 'nature-1', project: 'cooperation', clip: 1, x: 755, y: 105, width: 125, angle: 11 },
  { id: 'nature-2', project: 'cooperation', clip: 2, x: 920, y: 430, width: 95, angle: -12 },
  { id: 'nature-3', project: 'cooperation', clip: 3, x: 755, y: 443, width: 65, angle: 8 },
  { id: 'landscape', project: 'robust-steering', Component: RobustSteeringLandscape, x: 845, y: 175, width: 270, angle: 5 },
  { id: 'conversation', project: 'robust-steering', Component: RobustSteeringConversation, x: 450, y: 495, width: 350, angle: -3 },
  { id: 'signal', project: 'neural-field-manifold', Component: NeuralFieldSignal, x: 505, y: 100, width: 150, angle: -8 },
  { id: 'autoregressive', project: 'neural-field-manifold', Component: NeuralFieldAutoregressive, x: 785, y: 525, width: 310, angle: 5 },
  { id: 'manifold', project: 'neural-field-manifold', Component: NeuralFieldManifold, x: 65, y: 365, width: 380, angle: -6 },
].map(piece => ({ ...piece, width: piece.width * 1.2 }));

function Navigation({ selected, open, hover, tools, labels, omitAccepted }) {
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
        onFocus={() => setMenu(current => current === 'research' ? current : null)}
        onClick={() => setMenu('research')}
        onKeyDown={event => { if (event.key === 'ArrowDown') { event.preventDefault(); setMenu('research'); requestAnimationFrame(() => root.current.querySelector('.pb-dropdown a')?.focus()); } }}>{labels.research}<ChevronDown size={14} /></button>
      <div id="pb-research-menu" className="pb-dropdown" hidden={menu !== 'research'}>
        {Object.keys(projects).map(id => <a key={id} href={`#${id}`} aria-current={selected === id ? 'page' : undefined}
          onClick={event => { event.preventDefault(); setMenu(null); open(id, event.currentTarget); }}
          onPointerEnter={() => hover(id)} onFocus={() => hover(id)}>
          <span>{projects[id].title}</span><small>{publication(id, omitAccepted)}</small>
        </a>)}
      </div>
    </div>
    <div className="pb-nav-group" onPointerEnter={() => setMenu('packages')}>
      <button data-menu="packages" aria-expanded={menu === 'packages'} aria-controls="pb-packages-menu" onFocus={() => setMenu(current => current === 'packages' ? current : null)} onClick={() => setMenu('packages')}
        onKeyDown={event => { if (tools.length && event.key === 'ArrowDown') { event.preventDefault(); setMenu('packages'); requestAnimationFrame(() => root.current.querySelector('#pb-packages-menu a')?.focus()); } }}>{labels.packages}<ChevronDown size={14} /></button>
      <div id="pb-packages-menu" className={`pb-dropdown${tools.length ? '' : ' pb-packages'}`} hidden={menu !== 'packages'}>
        {tools.length ? tools.map(tool => <a key={tool.id} href={`#${tool.id}`} aria-current={selected === tool.id ? 'page' : undefined}
          onClick={event => { event.preventDefault(); setMenu(null); open(tool.id, event.currentTarget); }}
          onPointerEnter={() => hover(tool.id)} onFocus={() => hover(tool.id)}>
          <span><ProjectTitle project={tool} title={tool.navigationTitle} showLogo={tool.navigationLogo} /></span><small>{tool.year}</small>
        </a>) : <p>No packages listed yet.</p>}
      </div>
    </div>
    <a href="#about" aria-current={selected === 'about' ? 'page' : undefined} onPointerEnter={() => setMenu(null)} onFocus={() => setMenu(null)}
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

function ProjectDetail({ id, headingRef, portraitSlot, portraitFirst = false, tools, omitAccepted, aboutProfile = profile }) {
  const tool = tools.find(tool => tool.id === id);
  if (tool) return <ToolDetail tool={tool} headingRef={headingRef} />;
  if (id === 'about') return <article className="pb-about">
    {portraitFirst && <div className="pb-about-photo-slot" ref={portraitSlot} aria-hidden="true" />}
    <h2 tabIndex={-1} ref={headingRef}>{profile.name}</h2>
    {!portraitFirst && <div className="pb-about-photo-slot" ref={portraitSlot} aria-hidden="true" />}
    <p>{aboutProfile.role} <a href={aboutProfile.institutionUrl}>{aboutProfile.institution}</a>{aboutProfile.affiliation} {aboutProfile.statement}</p>
    <p>{profile.email}<br /><a href={profile.github}>{profile.githubLabel}</a></p>
  </article>;
  const content = projects[id];
  return <article className="pb-project" aria-labelledby="pb-project-heading">
    <h2 id="pb-project-heading" tabIndex={-1} ref={headingRef}>{content.title}</h2>
    <p className="pb-authors">{content.authors.map((author, index) => <span key={author}>{index > 0 ? ', ' : ''}<span className={author === profile.name ? 'pb-self' : undefined}>{author}</span>{content.equalContribution?.includes(author) && <sup>*</sup>}</span>)}</p>
    {content.equalContribution && <p className="pb-equal">* Equal contribution</p>}
    <div className="pb-project-links">
      <a href={content.url} target="_blank" rel="noopener noreferrer">{id === 'cooperation' ? <><img src={paper.natureLogo} alt="Nature" /> {content.year}</> : publication(id, omitAccepted)}<ArrowUpRight size={17} /></a>
      <a href={content.code} target="_blank" rel="noopener noreferrer"><img src={paper.githubLogo} alt="GitHub" /> code<ArrowUpRight size={17} /></a>
    </div>
    <div className="pb-project-media">
      {id === 'cooperation' && <NatureVideo />}
      {id === 'robust-steering' && <RobustSteeringAnimation />}
      {id === 'neural-field-manifold' && <NeuralFieldAnimation />}
    </div>
    <div className="pb-blurb">{content.blurb.split(/\n\s*\n/).map((paragraph,index) => <p key={index}>{paragraph}</p>)}</div>
    {content.abstractLicense && <p><small><a href={content.abstractLicense.url} rel="license">{content.abstractLicense.label}</a></small></p>}
  </article>;
}

export default function PrototypeB({ scenePieces = pieces, motion, variant = 'b', Portrait = TravelingPortrait, profileEntry = false, projectArrival = false, centeredStack = false, sidebarCopyright = false, aboutProfile = profile, tools = NO_TOOLS, navigationLabels = DEFAULT_NAVIGATION, className = '' }) {
  const [desktop, setDesktop] = useState(() => matchMedia('(min-width: 901px)').matches);
  const [selected, setSelected] = useState(() => selectedFromHash(tools));
  const [hovered, setHovered] = useState(null);
  const [transitioning, setTransitioning] = useState(false);
  const lastSelection = useRef(selected);
  const [paused, setPaused] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [reduced, setReduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [replay, setReplay] = useState(0);
  const [scale, setScale] = useState(1);
  const [viewport, setViewport] = useState(() => ({ width: innerWidth, height: innerHeight }));
  const scene = useRef(null);
  const canvas = useRef(null);
  const heading = useRef(null);
  const portraitSlot = useRef(null);
  const origin = useRef(null);
  const opener = useRef(null);
  const diagnostic = new URLSearchParams(location.search).get('benchmark');
  const active = hovered;
  const stackX = (24 - (viewport.width - 1200 * scale) / 2) / scale;
  const stackY = (viewport.height - 360 - 92) / scale;
  const arrival = useProjectArrival({ enabled: projectArrival, selected, desktop, canvas, scale, ...viewport });
  useOrbits(canvas, { desktop, paused, selected, hovered, reduced, replay, diagnostic, transitioning, motion });
  useLayoutEffect(() => {
    if (lastSelection.current === selected) { setTransitioning(false); return; }
    lastSelection.current = selected;
    setHovered(null);
    setTransitioning(!reduced);
    const timer = setTimeout(() => setTransitioning(false), reduced ? 0 : 750);
    return () => clearTimeout(timer);
  }, [selected, reduced]);
  function hover(project) {
    if (!selected && !transitioning) setHovered(project);
  }
  useEffect(() => {
    const media = matchMedia('(min-width: 901px)');
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const motionChange = () => { setReduced(motion.matches); setPaused(motion.matches); };
    const change = () => setDesktop(media.matches);
    const hash = () => { setSelected(selectedFromHash(tools)); setHovered(null); };
    media.addEventListener('change', change); window.addEventListener('hashchange', hash);
    motion.addEventListener('change', motionChange);
    return () => { media.removeEventListener('change', change); motion.removeEventListener('change', motionChange); window.removeEventListener('hashchange', hash); };
  }, [tools]);
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
    origin.current = target?.getBoundingClientRect();
    location.hash = id;
    setSelected(id); setHovered(null);
  }
  function close() {
    history.pushState(null, '', location.pathname + location.search);
    setSelected(null); setHovered(null);
    requestAnimationFrame(() => {
      document.querySelector('.pb-title')?.focus({ preventScroll: true });
    });
  }
  useEffect(() => {
    const pop = () => setSelected(selectedFromHash(tools));
    const escape = event => { if (event.key === 'Escape' && selected) close(); };
    window.addEventListener('popstate', pop); window.addEventListener('keydown', escape);
    return () => { window.removeEventListener('popstate', pop); window.removeEventListener('keydown', escape); };
  }, [selected, tools]);

  if (!desktop) return <main className="prototype-b pb-mobile">
    <img src={profile.photo} alt={profile.name} /><h1>{profile.name}</h1><p>{profile.statement}</p>
    <p>{profile.email}<br /><a href={profile.github}>{profile.githubLabel}</a></p>
    <p>Open on a computer for the full experience.</p>
  </main>;

  return <main className={`prototype-b${variant === 'b' ? '' : ` prototype-${variant}`}${className ? ` ${className}` : ''}`} data-selected={selected || ''}>
    <header className="pb-header" data-open={Boolean(selected)}>
      <Navigation selected={selected} open={open} hover={hover} tools={tools} labels={navigationLabels} omitAccepted={variant === 'd' || variant === 'e'} />
    </header>
    {selected && <aside className="pb-profile-dock" data-about={selected === 'about'}>
      <button className="pb-dock-back" onClick={close} aria-label="Return to composition" title="Return to composition"><ArrowLeft size={22} /></button>
      {selected !== 'about' && <div className="pb-identity">
        <h1>{profile.name}</h1>
        <span>{profile.email}</span><a href={profile.github} target="_blank" rel="noopener noreferrer">{profile.githubLabel}<ArrowUpRight size={13} /></a>
        {sidebarCopyright && <span className="pb-sidebar-copyright">&copy; 2026 {profile.name}</span>}
      </div>}
    </aside>}
    <section className="pb-stage" ref={scene} data-open={Boolean(selected)} aria-label="Interactive research composition" onPointerLeave={() => setHovered(null)}>
      <div className="pb-canvas" ref={canvas} style={{ '--pb-scale': scale }} data-open={Boolean(selected)} data-about={selected === 'about'} data-transitioning={transitioning} data-active={active || ''} data-paused={paused || Boolean(selected)}>
        <div className="pb-lettering" aria-hidden={Boolean(selected)} inert={selected ? true : undefined}>
          <h1 className="pb-title" tabIndex={-1}>{profileEntry
            ? <button className="pc-name-button" aria-label={`About ${profile.name}`} onPointerEnter={() => hover('about')} onPointerLeave={() => hover(null)} onFocus={() => hover('about')} onBlur={() => hover(null)} onClick={event => { if (!transitioning) open('about', event.currentTarget); }}><span>{profile.name.split(' ')[0]}</span><span>{profile.name.split(' ').slice(1).join(' ')}</span></button>
            : <><span>{profile.name.split(' ')[0]}</span><span>{profile.name.split(' ').slice(1).join(' ')}</span></>}</h1>
        </div>
        <NaturePreviewClock active={!paused && !selected && (!diagnostic || diagnostic === 'all' || diagnostic.startsWith('nature-'))} replay={replay}>{scenePieces.map((piece,index) => {
          const matching = active === piece.project;
          const arriving = projectArrival && selected === piece.project;
          const centered = centeredStack && selected && !arriving;
          const destination = arriving && arrival?.project === selected ? arrival : null;
          const settle = Boolean(active) && active !== 'about' && !matching;
          const factor = arriving ? Math.min(150 / piece.width, .55) : selected ? Math.min(110 / piece.width, .5) : 1;
          const x = arriving ? (destination?.x ?? 850) - piece.width * factor / 2
            : centered ? stackX + 104 / scale : selected ? stackX + 30 + index % 3 * 24 : piece.x + (settle ? (600 - piece.x - piece.width / 2) * .035 : 0);
          const y = arriving ? (destination?.y ?? 350) - (destination?.heights[piece.id] ?? piece.width) * factor / 2
            : centered ? stackY + 46 / scale : selected ? stackY + 10 + index % 4 * 15 : piece.y + (settle ? (350 - piece.y - piece.width / 2) * .035 : 0);
          const frozen = Boolean(selected) || Boolean(diagnostic && diagnostic !== 'all' && diagnostic !== piece.id);
          return <button key={piece.id} className="pb-fragment" data-piece={piece.id} data-project={piece.project} data-kind={piece.kind} data-matching={matching} data-arriving={arriving || undefined}
            style={{ width: piece.width, '--pb-angle': `${matching || selected ? 0 : piece.angle}deg`, transform: `translate(${x}px, ${y}px) rotate(${selected && !arriving ? (index - 4) * 5 : 0}deg) scale(${factor})${centered ? ' translate(-50%, -50%)' : ''}` }}
            aria-label={`${names[piece.project] || tools.find(tool => tool.id === piece.project)?.shortTitle}: ${piece.clip === undefined ? piece.id : `video ${piece.clip + 1}`}`}
            tabIndex={selected ? -1 : 0} aria-hidden={selected ? true : undefined} aria-controls="pb-reading"
            onClick={event => { if (!transitioning) selected ? close() : open(piece.project,event.currentTarget); }} onPointerEnter={() => hover(piece.project)} onPointerLeave={() => hover(null)} onFocus={() => hover(piece.project)} onBlur={() => hover(null)}>
            <div className="pb-orbit" data-project={piece.project} data-x={piece.x + piece.width / 2} data-y={piece.y + piece.width / 2} key={replay}>
              <div className="pb-tilt">
              <FreezePreview frozen={frozen}>
                {piece.Component ? <piece.Component compact active={!paused} maxFps={30} previewPadding={piece.id === 'landscape' ? 12 : 0} /> : <NaturePreview clip={piece.clip} poster={paper.clips[piece.clip].poster} />}
              </FreezePreview>
              </div>
            </div>
          </button>;
        })}</NaturePreviewClock>
      </div>
      {selected && <section id="pb-reading" className="pb-reading" key={selected}>
        <ProjectDetail id={selected} headingRef={heading} portraitSlot={portraitSlot} portraitFirst={profileEntry} tools={tools} omitAccepted={variant === 'd' || variant === 'e'} aboutProfile={aboutProfile} />
      </section>}
    </section>
    <Portrait selected={selected} slot={portraitSlot} origin={origin} viewport={viewport} reduced={reduced} close={close} open={open} hover={hover}
      canvas={canvas} scale={scale} paused={paused} hovered={hovered} replay={replay} transitioning={transitioning} />
    {(!sidebarCopyright || !selected) && <footer className="pb-footer">
      {!selected && !reduced && <div className="pb-scene-controls">
        <button onClick={() => setPaused(value => !value)} title={paused ? 'Play' : 'Pause'} aria-label={paused ? 'Play composition' : 'Pause composition'}>{paused ? <Play size={17} /> : <Pause size={17} />}</button>
        <button onClick={() => setReplay(value => value + 1)} title="Replay" aria-label="Replay composition"><RotateCcw size={17} /></button>
      </div>}
      {selected !== 'about' && <span>&copy; 2026 {profile.name}</span>}
    </footer>}
  </main>;
}
