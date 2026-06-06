import React, { useEffect, useMemo, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import {
  ArrowUpRight,
  BookOpen,
  Code2,
  Mail,
  Orbit,
  Sparkles
} from 'lucide-react';
import heroImage from '../images/hero.jpg';
import marlaxImage from '../images/marlax.gif';
import hmmImage from '../images/hmm.gif';
import '../stylesheet.css';

const contacts = [
  {
    label: 'rgs2151[at]columbia.eu',
    href: 'mailto:rgs2151@columbia.eu',
    icon: Mail
  },
  {
    label: 'GitHub',
    href: 'https://github.com/rgs2151',
    icon: Code2
  },
  {
    label: 'Google Scholar',
    href: 'https://scholar.google.com/citations?user=nN4ARxkAAAAJ&hl=en',
    icon: BookOpen
  }
];

const projects = [
  {
    image: marlaxImage,
    alt: 'Asymmetric Social Representations in the Prefrontal Cortex for Cooperative Behavior',
    title: 'Asymmetric Social Representations in the Prefrontal Cortex for Cooperative Behavior',
    href: 'https://doi.org/10.1101/2025.08.27.672249',
    authors:
      'Yuan Cheng, Yusi Chen, Myungji Kwak, Ross P. Kempner, Rudramani Singha, Jared Winslow, Runqi Liu, Umais Khan, Tessa Spangler, Alvi Khan, Talmo Pereira, Matthew Whiteway, Evan S. Schaffer, Nuttida Rungratsameetaweemana, Nan Yang, Herbert Zheng Wu',
    links: [
      { label: 'bioRxiv', href: 'https://doi.org/10.1101/2025.08.27.672249' },
      { label: 'code', href: 'https://github.com/NuttidaLab/MARLAX' }
    ],
    description:
      'We introduce a mouse paradigm to study cooperative behavior where stable leader-follower roles emerge during joint foraging. Using calcium imaging and optogenetic disruption, the study shows medial prefrontal cortex representations are role-specific and critical for cooperation. I developed the forward-modeling framework paired with multi-agent inverse reinforcement learning to decode latent value functions driving cooperative decisions.'
  },
  {
    image: hmmImage,
    alt: 'Bayesian Modeling Tutorial',
    title: 'Scaling Up Bayesian Models: Regressions, Mixtures, HMMs, and GLM-HMMs',
    href: 'https://art-of-neuron.github.io/',
    authors: 'Rudramani Singha',
    links: [
      { label: 'website', href: 'https://art-of-neuron.github.io/' },
      { label: 'code', href: 'https://github.com/art-of-neuron/art-of-neuron.github.io' }
    ],
    description:
      'This tutorial starts with intuitive Bayesian updates and builds to Hidden Markov Models and related latent-variable methods. It includes GLMs, input-driven Gaussian mixture models, and GLM-HMMs with comparisons between MCMC and EM estimation. Implementations cover PyMC, Stan, NumPyro, JAX, and Dynamax.'
  }
];

function NeuralField() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let width = 0;
    let height = 0;
    let frame = 0;
    let animationId = 0;
    let points = [];
    const pointer = { x: 0, y: 0, active: false };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const count = Math.max(44, Math.floor((width * height) / 15500));
      points = Array.from({ length: count }, (_, index) => ({
        x: ((index * 97) % Math.max(width, 1)) + Math.random() * 12,
        y: ((index * 53) % Math.max(height, 1)) + Math.random() * 12,
        vx: (Math.random() - 0.5) * 0.34,
        vy: (Math.random() - 0.5) * 0.34,
        pulse: Math.random() * Math.PI * 2
      }));
      if (prefersReducedMotion) {
        draw();
      }
    };

    const draw = () => {
      frame += 1;
      context.clearRect(0, 0, width, height);
      context.fillStyle = 'rgba(16, 16, 15, 0.82)';
      context.fillRect(0, 0, width, height);

      points.forEach((point) => {
        if (!prefersReducedMotion) {
          point.x += point.vx;
          point.y += point.vy;
          if (point.x < 0 || point.x > width) point.vx *= -1;
          if (point.y < 0 || point.y > height) point.vy *= -1;
        }

        if (pointer.active) {
          const dx = point.x - pointer.x;
          const dy = point.y - pointer.y;
          const distance = Math.hypot(dx, dy);
          if (distance < 160 && distance > 1) {
            point.x += (dx / distance) * 0.34;
            point.y += (dy / distance) * 0.34;
          }
        }
      });

      for (let i = 0; i < points.length; i += 1) {
        for (let j = i + 1; j < points.length; j += 1) {
          const a = points[i];
          const b = points[j];
          const distance = Math.hypot(a.x - b.x, a.y - b.y);
          if (distance < 126) {
            const alpha = (1 - distance / 126) * 0.46;
            context.strokeStyle = `rgba(38, 240, 207, ${alpha})`;
            context.lineWidth = 1;
            context.beginPath();
            context.moveTo(a.x, a.y);
            context.lineTo(b.x, b.y);
            context.stroke();
          }
        }
      }

      points.forEach((point, index) => {
        const pulse = 1.5 + Math.sin(frame * 0.025 + point.pulse) * 0.7;
        context.fillStyle = index % 3 === 0 ? '#d7ff3f' : index % 3 === 1 ? '#26f0cf' : '#ff4d2e';
        context.beginPath();
        context.arc(point.x, point.y, Math.max(1.2, pulse), 0, Math.PI * 2);
        context.fill();
      });

      if (!prefersReducedMotion) {
        animationId = requestAnimationFrame(draw);
      }
    };

    const handlePointerMove = (event) => {
      const rect = canvas.getBoundingClientRect();
      pointer.x = event.clientX - rect.left;
      pointer.y = event.clientY - rect.top;
      pointer.active = true;
    };

    const handlePointerLeave = () => {
      pointer.active = false;
    };

    resize();
    draw();
    window.addEventListener('resize', resize);
    canvas.addEventListener('pointermove', handlePointerMove);
    canvas.addEventListener('pointerleave', handlePointerLeave);

    return () => {
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('pointermove', handlePointerMove);
      canvas.removeEventListener('pointerleave', handlePointerLeave);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="neural-field" aria-hidden="true" />;
}

function ExternalLink({ href, children, className = '' }) {
  return (
    <a className={className} href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}

function ContactLink({ contact }) {
  const Icon = contact.icon;
  const external = !contact.href.startsWith('mailto:');
  const props = external ? { target: '_blank', rel: 'noopener noreferrer' } : {};

  return (
    <a className="contact-link" href={contact.href} title={contact.label} aria-label={contact.label} {...props}>
      <Icon aria-hidden="true" size={18} strokeWidth={1.8} />
      <span>{contact.label}</span>
    </a>
  );
}

function AuthorLine({ text }) {
  const name = 'Rudramani Singha';
  if (!text.includes(name)) {
    return (
      <p className="paper-authors">
        <strong>{text}</strong>
      </p>
    );
  }

  const parts = text.split(name);
  return (
    <p className="paper-authors">
      {parts[0]}
      <strong>{name}</strong>
      {parts.slice(1).join(name)}
    </p>
  );
}

function ProjectCard({ project, index }) {
  const projectNumber = String(index + 1).padStart(2, '0');
  const accents = ['#26f0cf', '#ff4d2e'];

  return (
    <article className="project-card" style={{ '--project-accent': accents[index % accents.length] }}>
      <div className="project-media">
        <img src={project.image} alt={project.alt} />
      </div>
      <div className="project-copy">
        <span className="project-number" aria-hidden="true">
          {projectNumber}
        </span>
        <ExternalLink href={project.href} className="project-title">
          <span>{project.title}</span>
          <ArrowUpRight aria-hidden="true" size={22} strokeWidth={1.8} />
        </ExternalLink>
        <AuthorLine text={project.authors} />
        <p className="paper-links">
          <em>links:</em>{' '}
          {project.links.map((link) => (
            <React.Fragment key={link.href}>
              [
              <ExternalLink href={link.href}>{link.label}</ExternalLink>
              ]{' '}
            </React.Fragment>
          ))}
        </p>
        <p className="project-description">{project.description}</p>
      </div>
    </article>
  );
}

function App() {
  const projectList = useMemo(() => projects, []);

  return (
    <main className="site">
      <section className="hero" aria-label="Rudramani Singha">
        <NeuralField />
        <img className="hero-image" alt="Rudramani Singha profile photo" src={heroImage} />
        <div className="hero-scrim" aria-hidden="true" />
        <div className="hero-content">
          <div className="hero-mark" aria-hidden="true">
            <Orbit size={28} strokeWidth={1.6} />
          </div>
          <h1>Rudramani Singha</h1>
          <p className="intro-copy">
            I am a Data Scientist at the{' '}
            <ExternalLink href="https://memorylongevity.org/">Program in Memory Longevity</ExternalLink>, UTSW. I
            build probabilistic models to understand the brain.
          </p>
          <nav className="contact-strip" aria-label="Contact links">
            {contacts.map((contact) => (
              <ContactLink contact={contact} key={contact.href} />
            ))}
          </nav>
        </div>
      </section>

      <section className="projects" aria-labelledby="selected-projects">
        <div className="section-heading">
          <Sparkles aria-hidden="true" size={20} strokeWidth={1.8} />
          <h2 id="selected-projects">Selected Projects</h2>
        </div>
        <div className="project-list">
          {projectList.map((project, index) => (
            <ProjectCard project={project} index={index} key={project.title} />
          ))}
        </div>
      </section>

      <footer className="site-footer">
        <p>&copy; 2026 Rudramani Singha</p>
      </footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
