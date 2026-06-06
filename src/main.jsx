import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import figlet from 'figlet';
import heroImage from '../images/hero.jpg';
import marlaxImage from '../images/marlax.gif';
import hmmImage from '../images/hmm.gif';
import '../stylesheet.css';

const ASCII_RAMP = ' .:-=+*#%@';
const DONUT_RAMP = '.,-~:;=!*#$@';

const contacts = [
  {
    label: 'rgs2151[at]columbia.eu',
    href: 'mailto:rgs2151@columbia.eu'
  },
  {
    label: 'GitHub',
    href: 'https://github.com/rgs2151'
  },
  {
    label: 'Google Scholar',
    href: 'https://scholar.google.com/citations?user=nN4ARxkAAAAJ&hl=en'
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

function useFiglet(text) {
  const [output, setOutput] = useState(text);

  useEffect(() => {
    let mounted = true;
    figlet.text(text, { font: 'Slant' }, (error, result) => {
      if (!mounted) return;
      setOutput(error ? text : result);
    });
    return () => {
      mounted = false;
    };
  }, [text]);

  return output;
}

function luminance(red, green, blue) {
  return 0.299 * red + 0.587 * green + 0.114 * blue;
}

function frameToAscii(image, columns, rows, invert = false) {
  const canvas = document.createElement('canvas');
  canvas.width = columns;
  canvas.height = rows;
  const context = canvas.getContext('2d');
  context.drawImage(image, 0, 0, columns, rows);
  const { data } = context.getImageData(0, 0, columns, rows);
  let output = '';

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const index = (y * columns + x) * 4;
      const alpha = data[index + 3] / 255;
      const value = alpha === 0 ? 0 : luminance(data[index], data[index + 1], data[index + 2]);
      const normalized = invert ? 1 - value / 255 : value / 255;
      const rampIndex = Math.min(ASCII_RAMP.length - 1, Math.max(0, Math.floor(normalized * ASCII_RAMP.length)));
      output += ASCII_RAMP[rampIndex];
    }
    output += '\n';
  }

  return output;
}

function AsciiMedia({ src, alt, columns = 76, rows = 34, cadence = 90, invert = false }) {
  const [ascii, setAscii] = useState('');
  const imageRef = useRef(null);

  useEffect(() => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.src = src;
    imageRef.current = image;
    let frame = 0;
    let intervalId = 0;

    const render = () => {
      if (!image.complete || image.naturalWidth === 0) return;
      try {
        setAscii(frameToAscii(image, columns, rows, invert));
        frame += 1;
      } catch {
        clearInterval(intervalId);
      }
    };

    image.addEventListener('load', render);
    intervalId = window.setInterval(render, cadence);

    return () => {
      image.removeEventListener('load', render);
      clearInterval(intervalId);
    };
  }, [src, columns, rows, cadence, invert]);

  return (
    <figure className="ascii-media" aria-label={alt}>
      <pre aria-hidden="true">{ascii || 'loading ascii stream...'}</pre>
      <figcaption>{alt}</figcaption>
    </figure>
  );
}

function renderDonut(width, height, angleA, angleB) {
  const output = Array(width * height).fill(' ');
  const zBuffer = Array(width * height).fill(0);
  const cosA = Math.cos(angleA);
  const sinA = Math.sin(angleA);
  const cosB = Math.cos(angleB);
  const sinB = Math.sin(angleB);
  const radiusOne = 1;
  const radiusTwo = 2;
  const distance = 5;
  const scale = (width * distance * 3) / (8 * (radiusOne + radiusTwo));

  for (let theta = 0; theta < Math.PI * 2; theta += 0.07) {
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);

    for (let phi = 0; phi < Math.PI * 2; phi += 0.02) {
      const cosphi = Math.cos(phi);
      const sinphi = Math.sin(phi);
      const circle = radiusTwo + radiusOne * costheta;

      const x =
        circle * (cosB * cosphi + sinA * sinB * sinphi) - radiusOne * cosA * sinB * sintheta;
      const y =
        circle * (sinB * cosphi - sinA * cosB * sinphi) + radiusOne * cosA * cosB * sintheta;
      const z = distance + cosA * circle * sinphi + radiusOne * sinA * sintheta;
      const inverseZ = 1 / z;
      const xp = Math.floor(width / 2 + scale * inverseZ * x);
      const yp = Math.floor(height / 2 - scale * 0.52 * inverseZ * y);
      const luminanceValue =
        cosphi * costheta * sinB -
        cosA * costheta * sinphi -
        sinA * sintheta +
        cosB * (cosA * sintheta - costheta * sinA * sinphi);

      if (luminanceValue > 0 && xp >= 0 && xp < width && yp >= 0 && yp < height) {
        const outputIndex = xp + width * yp;
        if (inverseZ > zBuffer[outputIndex]) {
          zBuffer[outputIndex] = inverseZ;
          const shade = Math.min(DONUT_RAMP.length - 1, Math.floor(luminanceValue * 7));
          output[outputIndex] = DONUT_RAMP[shade];
        }
      }
    }
  }

  let frame = '';
  for (let row = 0; row < height; row += 1) {
    frame += output.slice(row * width, row * width + width).join('') + '\n';
  }
  return frame;
}

function AsciiDonut() {
  const [frame, setFrame] = useState('');

  useEffect(() => {
    let angleA = 0;
    let angleB = 0;
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const tick = () => {
      setFrame(renderDonut(64, 28, angleA, angleB));
      angleA += 0.07;
      angleB += 0.035;
    };
    tick();
    if (prefersReducedMotion) return undefined;
    const intervalId = window.setInterval(tick, 52);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="donut-terminal" aria-hidden="true">
      <pre>{frame}</pre>
    </div>
  );
}

function ExternalLink({ href, children, className = '' }) {
  return (
    <a className={className} href={href} target="_blank" rel="noopener noreferrer">
      {children}
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

function TerminalLine({ label, children }) {
  return (
    <p className="terminal-line">
      <span>{label}</span>
      {children}
    </p>
  );
}

function ProjectBlock({ project, index }) {
  return (
    <article className="project-block">
      <div className="project-output">
        <AsciiMedia src={project.image} alt={project.alt} columns={72} rows={28} cadence={index === 0 ? 70 : 95} />
      </div>
      <div className="project-terminal">
        <TerminalLine label={`PROJECT_${String(index + 1).padStart(2, '0')}`}>
          <ExternalLink href={project.href}>{project.title}</ExternalLink>
        </TerminalLine>
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
  const title = useFiglet('Rudramani Singha');
  const projectList = useMemo(() => projects, []);

  return (
    <main className="site">
      <section className="hero" aria-label="Rudramani Singha">
        <div className="scanline" aria-hidden="true" />
        <div className="hero-grid">
          <div className="hero-terminal">
            <p className="boot-line">root@singha:~$ ./reverse_engineer_brain --mode=probabilistic</p>
            <pre className="ascii-title">{title}</pre>
            <TerminalLine label="STATUS">
              I am a Data Scientist at the{' '}
              <ExternalLink href="https://memorylongevity.org/">Program in Memory Longevity</ExternalLink>, UTSW. I
              build probabilistic models to understand the brain.
            </TerminalLine>
            <nav className="contact-strip" aria-label="Contact links">
              {contacts.map((contact) => (
                <a
                  className="contact-link"
                  href={contact.href}
                  key={contact.href}
                  target={contact.href.startsWith('mailto:') ? undefined : '_blank'}
                  rel={contact.href.startsWith('mailto:') ? undefined : 'noopener noreferrer'}
                >
                  [{contact.label}]
                </a>
              ))}
            </nav>
          </div>
          <div className="hero-ascii">
            <AsciiMedia src={heroImage} alt="Rudramani Singha profile photo" columns={64} rows={36} cadence={180} />
          </div>
          <AsciiDonut />
        </div>
      </section>

      <section className="projects" aria-labelledby="selected-projects">
        <div className="section-heading">
          <pre aria-hidden="true">{'//=============================================================='}</pre>
          <h2 id="selected-projects">Selected Projects</h2>
          <pre aria-hidden="true">{'//=============================================================='}</pre>
        </div>
        <div className="project-list">
          {projectList.map((project, index) => (
            <ProjectBlock project={project} index={index} key={project.title} />
          ))}
        </div>
      </section>

      <footer className="site-footer">
        <p>(c) 2026 Rudramani Singha</p>
      </footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
