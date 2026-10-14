import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, X } from 'lucide-react';
import { profile, paper, robustSteering, neuralFieldManifold } from '../../content/site';
import ParticleField from '../../shared/ParticleField';
import NeuralFieldAnimation, { NeuralFieldSignal, NeuralFieldAutoregressive, NeuralFieldManifold } from '../../shared/NeuralFieldAnimation';
import ProjectVideo from '../../shared/ProjectVideo';
import RobustSteeringAnimation, { RobustSteeringLandscape, RobustSteeringConversation } from '../../shared/RobustSteeringAnimation';
import './styles.css';

const projectContent = { biological: paper, artificial: robustSteering, manifold: neuralFieldManifold };

function ProjectView({ project, onClose }) {
  if (project === 'about') {
    return <div className="pa-about">
      <img src={profile.photo} alt={profile.name} />
      <div><h2>{profile.name}</h2>
        <p>{profile.role} <a href={profile.institutionUrl}>{profile.institution}</a>{profile.affiliation} {profile.statement}</p>
        <p>{profile.email}<br /><a href={profile.github}>{profile.githubLabel}</a></p>
      </div>
    </div>;
  }
  const content = projectContent[project];
  return <article className={`pa-paper pa-paper-${project}`}>
    <h2>{content.title}</h2>
    {project === 'biological' && <ProjectVideo src={paper.strip} poster={paper.stripPoster} />}
    {project === 'artificial' && <RobustSteeringAnimation onClose={onClose} />}
    {project === 'manifold' && <NeuralFieldAnimation />}
    <div className="pa-paper-links">
      <a href={content.url} target="_blank" rel="noopener noreferrer">{project === 'biological' ? <img src={paper.natureLogo} alt="Nature" /> : 'Project'} {content.year}<ArrowUpRight size={18} /></a>
      <a href={content.code} target="_blank" rel="noopener noreferrer"><img src={paper.githubLogo} alt="GitHub" /> code<ArrowUpRight size={18} /></a>
    </div>
    <p className="pa-authors">{content.authors.map((author, index) => <span key={author}>
      {index > 0 ? ', ' : ''}<span className={author === profile.name ? 'pa-self' : undefined}>{author}</span>{content.equalContribution?.includes(author) && <sup>*</sup>}
    </span>)}</p>
    {content.equalContribution && <p className="pa-equal">* Equal contribution</p>}
    <p className="pa-blurb">{content.blurb}</p>
    {content.abstractLicense && <p><small><a href={content.abstractLicense.url} rel="license">{content.abstractLicense.label}</a></small></p>}
  </article>;
}

export default function PrototypeA() {
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [scale, setScale] = useState(1);
  const [desktop, setDesktop] = useState(() => matchMedia('(min-width: 741px)').matches);
  const stageRef = useRef(null);
  const dialogRef = useRef(null);
  const closeRef = useRef(null);
  const active = selected || hovered;

  useEffect(() => {
    const media = matchMedia('(min-width: 741px)');
    const update = () => setDesktop(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

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
          <button className="pa-model" onClick={() => openProject('artificial')} aria-label={robustSteering.shortTitle} aria-haspopup="dialog">
            {desktop && <RobustSteeringLandscape compact active={!selected} />}
            <span>{robustSteering.shortTitle}<ArrowUpRight size={24} /></span>
          </button>
          {desktop && <button className="pa-steering-conversation" onClick={() => openProject('artificial')} aria-label="Open Robust steering: conversation" aria-haspopup="dialog">
            <RobustSteeringConversation compact active={!selected} />
          </button>}
        </div>
        <div className="pa-manifold" onMouseEnter={() => setHovered('manifold')} onFocus={() => setHovered('manifold')}
          onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setHovered(null); }}>
          {desktop && <div className="pa-manifold-fragments" role="group" aria-label="NeuralFieldManifold animations">
            <button className="pa-manifold-fragment pa-manifold-signal" onClick={() => openProject('manifold')} aria-label="Open NeuralFieldManifold: signal" aria-haspopup="dialog">
              <NeuralFieldSignal compact active={!selected} />
            </button>
            <button className="pa-manifold-fragment pa-manifold-autoregressive" onClick={() => openProject('manifold')} aria-label="Open NeuralFieldManifold: autoregressive state" aria-haspopup="dialog">
              <NeuralFieldAutoregressive compact active={!selected} />
            </button>
            <button className="pa-manifold-fragment pa-manifold-trajectory" onClick={() => openProject('manifold')} aria-label="Open NeuralFieldManifold: manifold" aria-haspopup="dialog">
              <NeuralFieldManifold compact active={!selected} />
            </button>
          </div>}
          <button className="pa-manifold-project" onClick={() => openProject('manifold')} aria-label={neuralFieldManifold.title} aria-haspopup="dialog">
            <span>NeuralFieldManifold<ArrowUpRight size={24} /></span>
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
      onClick={event => { if (event.target === event.currentTarget) closeProject(); }} aria-label={selected === 'about' ? 'About Rudramani Singha' : projectContent[selected]?.title || 'Project'}>
      <div className="pa-dialog-inner">
        <button className="pa-close" ref={closeRef} onClick={closeProject} aria-label="Close project" title="Close project"><X size={24} /></button>
        {selected && <ProjectView key={selected} project={selected} onClose={closeProject} />}
      </div>
    </dialog>
  </main>;
}
