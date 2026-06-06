import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import figlet from 'figlet';
import { decompressFrames, parseGIF } from 'gifuct-js';
import heroImage from '../images/hero.jpg';
import marlaxImage from '../images/marlax.gif';
import hmmImage from '../images/hmm.gif';
import '../stylesheet.css';

const ASCII_RAMP = ' .,:;irsXA253hMHGS#9B&@';
const GLYPHS = '01._:/\\\\|+-=*#';
const CHAR_ASPECT = 0.58;

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
      if (mounted) setOutput(error ? text : result);
    });
    return () => {
      mounted = false;
    };
  }, [text]);

  return output;
}

function getRows(width, height, columns) {
  return Math.max(10, Math.round(columns * (height / width) * CHAR_ASPECT));
}

function luminance(red, green, blue) {
  return 0.299 * red + 0.587 * green + 0.114 * blue;
}

function canvasToAscii(sourceCanvas, columns, rows, invert = false) {
  const canvas = document.createElement('canvas');
  canvas.width = columns;
  canvas.height = rows;
  const context = canvas.getContext('2d');
  context.fillStyle = '#000';
  context.fillRect(0, 0, columns, rows);
  context.drawImage(sourceCanvas, 0, 0, columns, rows);

  const { data } = context.getImageData(0, 0, columns, rows);
  let output = '';

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const index = (y * columns + x) * 4;
      const alpha = data[index + 3] / 255;
      const value = alpha === 0 ? 0 : luminance(data[index], data[index + 1], data[index + 2]);
      const normalized = invert ? 1 - value / 255 : value / 255;
      const rampIndex = Math.min(
        ASCII_RAMP.length - 1,
        Math.max(0, Math.floor(normalized * (ASCII_RAMP.length - 1)))
      );
      output += ASCII_RAMP[rampIndex];
    }
    output += '\n';
  }

  return output;
}

function imageToAscii(image, columns, invert = false) {
  const rows = getRows(image.naturalWidth, image.naturalHeight, columns);
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = image.naturalWidth;
  sourceCanvas.height = image.naturalHeight;
  sourceCanvas.getContext('2d').drawImage(image, 0, 0);
  return { text: canvasToAscii(sourceCanvas, columns, rows, invert), rows };
}

async function gifToAsciiFrames(src, columns, invert = false) {
  const response = await fetch(src);
  const buffer = await response.arrayBuffer();
  const gif = parseGIF(buffer);
  const frames = decompressFrames(gif, true);
  const width = gif.lsd.width;
  const height = gif.lsd.height;
  const rows = getRows(width, height, columns);
  const sourceCanvas = document.createElement('canvas');
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const context = sourceCanvas.getContext('2d');
  context.fillStyle = '#000';
  context.fillRect(0, 0, width, height);

  const usableFrames = frames.filter((_, index) => index % 2 === 0).slice(0, 90);

  return usableFrames.map((frame) => {
    const imageData = new ImageData(new Uint8ClampedArray(frame.patch), frame.dims.width, frame.dims.height);
    context.putImageData(imageData, frame.dims.left, frame.dims.top);
    return {
      text: canvasToAscii(sourceCanvas, columns, rows, invert),
      delay: Math.max(42, frame.delay || 80)
    };
  });
}

function AsciiMedia({ src, alt, columns = 74, cadence = 90, invert = false, animated = false }) {
  const [frames, setFrames] = useState([]);
  const [index, setIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      if (animated) {
        const decoded = await gifToAsciiFrames(src, columns, invert);
        if (!cancelled) setFrames(decoded);
        return;
      }

      const image = new Image();
      image.crossOrigin = 'anonymous';
      image.src = src;
      image.addEventListener('load', () => {
        if (!cancelled) setFrames([{ ...imageToAscii(image, columns, invert), delay: cadence }]);
      });
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [src, columns, cadence, invert, animated]);

  useEffect(() => {
    if (frames.length <= 1) return undefined;
    const currentDelay = frames[index]?.delay || cadence;
    const timeoutId = window.setTimeout(() => {
      setIndex((value) => (value + 1) % frames.length);
    }, currentDelay);
    return () => clearTimeout(timeoutId);
  }, [frames, index, cadence]);

  const activeFrame = frames[index]?.text || 'loading ascii stream...';

  return (
    <figure className="ascii-media" aria-label={alt}>
      <pre aria-hidden="true">{activeFrame}</pre>
      <figcaption>{alt}</figcaption>
    </figure>
  );
}

function SignalField() {
  const [frame, setFrame] = useState('');

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let tick = 0;

    const render = () => {
      const lines = [];
      for (let row = 0; row < 16; row += 1) {
        let line = '';
        for (let column = 0; column < 78; column += 1) {
          const value = Math.sin((column + tick) * 0.27) + Math.cos((row * 4 - tick) * 0.19);
          const charIndex = Math.abs(Math.floor((value + 2) * 4 + row + column + tick)) % GLYPHS.length;
          line += GLYPHS[charIndex];
        }
        lines.push(line);
      }
      setFrame(lines.join('\n'));
      tick += 1;
    };

    render();
    if (prefersReducedMotion) return undefined;
    const intervalId = window.setInterval(render, 90);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="signal-field" aria-hidden="true">
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
        <AsciiMedia src={project.image} alt={project.alt} columns={index === 0 ? 66 : 76} animated />
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
        <div className="hero-grid">
          <div className="hero-terminal">
            <p className="boot-line">~/singha.io $ model --probabilistic --brain</p>
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
            <AsciiMedia src={heroImage} alt="Rudramani Singha profile photo" columns={64} cadence={180} />
          </div>
          <SignalField />
        </div>
      </section>

      <section className="projects" aria-labelledby="selected-projects">
        <div className="section-heading">
          <pre aria-hidden="true">{'//------------------------------'}</pre>
          <h2 id="selected-projects">Selected Projects</h2>
          <pre aria-hidden="true">{'------------------------------//'}</pre>
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
