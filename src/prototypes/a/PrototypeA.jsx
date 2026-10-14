import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, X } from 'lucide-react';
import { profile, paper } from '../../content/site';
import ParticleField from '../../shared/ParticleField';
import './styles.css';

function ProjectView({ project }) {
  const [steering, setSteering] = useState(0);
  if (project === 'about') {
    return <div className="pa-about">
      <img src={profile.photo} alt={profile.name} />
      <div><h2>{profile.name}</h2>
        <p>{profile.role} <a href={profile.institutionUrl}>{profile.institution}</a>{profile.affiliation} {profile.statement}</p>
        <p>{profile.email}<br /><a href={profile.github}>{profile.githubLabel}</a></p>
      </div>
    </div>;
  }
  if (project === 'artificial') {
    return <div className="pa-steering-view">
      <h2>LLM steering</h2>
      <ParticleField steering={steering} />
      <label>Steering <input type="range" min="-1" max="1" step="0.01" value={steering}
        onChange={event => setSteering(Number(event.target.value))} /></label>
    </div>;
  }
  return <article className="pa-paper">
    <h2>{paper.title}</h2>
    <video src={paper.strip} poster={paper.stripPoster} controls muted loop playsInline preload="metadata" />
    <div className="pa-paper-links">
      <a href={paper.url} target="_blank" rel="noopener noreferrer"><img src={paper.natureLogo} alt="Nature" /> {paper.year}<ArrowUpRight size={18} /></a>
      <a href={paper.code} target="_blank" rel="noopener noreferrer"><img src={paper.githubLogo} alt="GitHub" /> code<ArrowUpRight size={18} /></a>
    </div>
    <p className="pa-authors">{paper.authors.map((author, index) => <span key={author}>
      {index > 0 ? ', ' : ''}<span className={author === profile.name ? 'pa-self' : undefined}>{author}</span>
    </span>)}</p>
    <p>{paper.blurb}</p>
  </article>;
}

export default function PrototypeA() {
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [scale, setScale] = useState(1);
  const stageRef = useRef(null);
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const active = selected || hovered;

  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      setScale(Math.min(entry.contentRect.width / 1140, entry.contentRect.height / 700, 1.25));
    });
    observer.observe(stageRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (selected && !dialog.open) {
      dialog.showModal();
      closeRef.current.focus();
    } else if (!selected && dialog.open) dialog.close();
  }, [selected]);

  function openProject(project) {
    setSelected(project);
    setHovered(project);
  }

  function closeProject() {
    setSelected(null);
    setHovered(null);
  }

  return <main className="prototype-a">
    <header className="pa-identity">
      <button className="pa-portrait" onClick={() => openProject('about')} title="About Rudramani Singha" aria-label="About Rudramani Singha">
        <img src={profile.photo} alt={profile.name} />
      </button>
      <div><span>{profile.name}</span><span>{profile.email}</span>
        <a href={profile.github} target="_blank" rel="noopener noreferrer">{profile.githubLabel}<ArrowUpRight size={13} /></a>
      </div>
    </header>

    <section className="pa-stage" ref={stageRef} aria-label="Interactive research composition" onMouseLeave={() => setHovered(null)}>
      <div className={`pa-composition ${active ? `pa-active-${active}` : ''}`} style={{ '--scene-scale': scale }}>
        <div className="pa-backbone" aria-hidden="true"><ParticleField variant="folded" /></div>
        <div className="pa-words" aria-hidden="true">
          {paper.title.match(/leader|follower/g).map(word => <span key={word}>{word}</span>)}
        </div>
        <div className="pa-biological" onMouseEnter={() => setHovered('biological')}
          onFocus={() => setHovered('biological')}
          onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setHovered(null); }}>
          {paper.clips.map((clip, index) => <button className={`pa-fragment pa-fragment-${index}`} key={clip.src}
            onClick={() => openProject('biological')} aria-label={`${paper.title}, panel ${index + 1}`} aria-haspopup="dialog">
            <img src={clip.poster} alt={`Supplementary video panel ${index + 1}`} />
          </button>)}
          <button className="pa-project-label" onClick={() => openProject('biological')} aria-haspopup="dialog">
            {paper.title}<ArrowUpRight size={24} />
          </button>
        </div>
        <div className="pa-artificial" onMouseEnter={() => setHovered('artificial')}
          onFocus={() => setHovered('artificial')} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setHovered(null); }}>
          <button className="pa-model" onClick={() => openProject('artificial')} aria-label="LLM steering" aria-haspopup="dialog">
            <div className="pa-model-field"><ParticleField steering={active === 'artificial' ? 0.8 : 0} /></div>
            <div className="pa-model-layer pa-model-layer-one"><ParticleField steering={-0.4} /></div>
            <div className="pa-model-layer pa-model-layer-two"><ParticleField steering={0.4} /></div>
            <span>LLM<br />steering<ArrowUpRight size={28} /></span>
          </button>
        </div>
        <div className="pa-paper-stamp" aria-hidden="true"><img src={paper.natureLogo} alt="" /><span>{paper.year}</span></div>
      </div>
    </section>

    <footer className="pa-footer"><button onClick={() => openProject('about')}>About</button><span>&copy; 2026 {profile.name}</span></footer>
    <section className="pa-mobile">
      <img src={profile.photo} alt={profile.name} /><h1>{profile.name}</h1><p>{profile.statement}</p>
      <p>{profile.email}<br /><a href={profile.github}>{profile.githubLabel}</a></p>
      <p>Open on a computer for the full experience.</p>
    </section>

    <dialog className="pa-dialog" ref={dialogRef} onCancel={closeProject} onClose={closeProject}
      onClick={event => { if (event.target === event.currentTarget) closeProject(); }} aria-label={selected === 'biological' ? paper.title : selected === 'about' ? 'About Rudramani Singha' : 'LLM steering'}>
      <div className="pa-dialog-inner">
        <button className="pa-close" ref={closeRef} onClick={closeProject} aria-label="Close project" title="Close project"><X size={24} /></button>
        {selected && <ProjectView key={selected} project={selected} />}
      </div>
    </dialog>
  </main>;
}
