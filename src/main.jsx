import React, { useEffect, useMemo, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import '../stylesheet.css';

const MONO_FONT = 'ui-monospace, SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace';

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
    type: 'marlax',
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
    type: 'hmm',
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

function useCanvas(draw, deps = []) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let frame = 0;
    let animationId = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(context, rect.width, rect.height, frame);
    };

    const render = () => {
      const rect = canvas.getBoundingClientRect();
      draw(context, rect.width, rect.height, frame);
      frame += 1;
      if (!prefersReducedMotion) animationId = requestAnimationFrame(render);
    };

    resize();
    render();
    window.addEventListener('resize', resize);

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationId);
    };
  }, deps);

  return canvasRef;
}

function clearCanvas(context, width, height) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#050505';
  context.fillRect(0, 0, width, height);
}

function drawGrid(context, x, y, size, cells) {
  context.save();
  context.strokeStyle = 'rgba(255,255,255,0.16)';
  context.lineWidth = 1;
  for (let index = 0; index <= cells; index += 1) {
    const position = x + (index / cells) * size;
    context.beginPath();
    context.moveTo(position, y);
    context.lineTo(position, y + size);
    context.stroke();
    context.beginPath();
    context.moveTo(x, y + (index / cells) * size);
    context.lineTo(x + size, y + (index / cells) * size);
    context.stroke();
  }
  context.strokeStyle = 'rgba(255,255,255,0.9)';
  context.lineWidth = 1.4;
  context.strokeRect(x, y, size, size);
  context.restore();
}

function gridPoint(originX, originY, size, gridSize, point) {
  return {
    x: originX + (point.x / (gridSize - 1)) * size,
    y: originY + size - (point.y / (gridSize - 1)) * size
  };
}

function interpolatePath(path, progress) {
  const wrapped = progress % path.length;
  const index = Math.floor(wrapped);
  const next = (index + 1) % path.length;
  const amount = wrapped - index;
  return {
    x: path[index].x + (path[next].x - path[index].x) * amount,
    y: path[index].y + (path[next].y - path[index].y) * amount,
    previous: path[index],
    next: path[next]
  };
}

function drawMouse(context, x, y, angle, scale, alpha = 1) {
  context.save();
  context.translate(x, y);
  context.rotate(angle);
  context.strokeStyle = `rgba(255,255,255,${alpha})`;
  context.fillStyle = '#050505';
  context.lineWidth = Math.max(1.4, scale * 0.065);
  context.lineCap = 'round';
  context.lineJoin = 'round';

  context.beginPath();
  context.ellipse(0, 0, scale * 0.35, scale * 0.56, 0, 0, Math.PI * 2);
  context.stroke();

  context.beginPath();
  context.arc(0, -scale * 0.58, scale * 0.28, 0, Math.PI * 2);
  context.stroke();

  context.beginPath();
  context.arc(-scale * 0.19, -scale * 0.78, scale * 0.12, 0, Math.PI * 2);
  context.stroke();
  context.beginPath();
  context.arc(scale * 0.19, -scale * 0.78, scale * 0.12, 0, Math.PI * 2);
  context.stroke();

  context.beginPath();
  context.moveTo(-scale * 0.08, -scale * 0.84);
  context.lineTo(0, -scale * 1.02);
  context.lineTo(scale * 0.08, -scale * 0.84);
  context.stroke();

  context.beginPath();
  context.arc(-scale * 0.09, -scale * 0.62, scale * 0.018, 0, Math.PI * 2);
  context.fillStyle = `rgba(255,255,255,${alpha})`;
  context.fill();
  context.beginPath();
  context.arc(scale * 0.09, -scale * 0.62, scale * 0.018, 0, Math.PI * 2);
  context.fill();

  context.beginPath();
  context.moveTo(0, scale * 0.52);
  context.bezierCurveTo(-scale * 0.18, scale * 0.82, -scale * 0.42, scale * 0.9, -scale * 0.56, scale * 1.12);
  context.stroke();
  context.restore();
}

function MouseSketchCanvas() {
  const pathA = useMemo(
    () => [
      { x: 2, y: 2 },
      { x: 5, y: 2 },
      { x: 8, y: 3 },
      { x: 9, y: 6 },
      { x: 7, y: 8 },
      { x: 4, y: 8 },
      { x: 2, y: 6 },
      { x: 2, y: 2 }
    ],
    []
  );
  const pathB = useMemo(
    () => [
      { x: 8, y: 8 },
      { x: 5, y: 8 },
      { x: 3, y: 7 },
      { x: 1, y: 5 },
      { x: 3, y: 3 },
      { x: 6, y: 3 },
      { x: 8, y: 5 },
      { x: 8, y: 8 }
    ],
    []
  );

  const canvasRef = useCanvas((context, width, height, frame) => {
    clearCanvas(context, width, height);
    const size = Math.min(width, height) * 0.78;
    const originX = (width - size) / 2;
    const originY = (height - size) / 2;
    const gridSize = 11;
    drawGrid(context, originX, originY, size, gridSize - 1);

    const t = frame * 0.018;
    const positions = [interpolatePath(pathA, t), interpolatePath(pathB, t + 2.8)];
    const cell = size / (gridSize - 1);

    context.save();
    context.strokeStyle = 'rgba(255,255,255,0.44)';
    context.lineWidth = 1.5;
    positions.forEach((position) => {
      const point = gridPoint(originX, originY, size, gridSize, position);
      context.beginPath();
      context.arc(point.x, point.y, cell * 0.78, 0, Math.PI * 2);
      context.stroke();
    });
    context.restore();

    const rewardCycle = Math.floor(frame / 130) % 4;
    const rewards = [
      { x: 5, y: 10 },
      { x: 10, y: 5 },
      { x: 5, y: 0 },
      { x: 0, y: 5 }
    ];
    const reward = gridPoint(originX, originY, size, gridSize, rewards[rewardCycle]);
    context.save();
    context.strokeStyle = 'rgba(255,255,255,0.82)';
    context.lineWidth = 1.6;
    context.beginPath();
    context.moveTo(reward.x, reward.y - cell * 0.35);
    context.lineTo(reward.x + cell * 0.35, reward.y);
    context.lineTo(reward.x, reward.y + cell * 0.35);
    context.lineTo(reward.x - cell * 0.35, reward.y);
    context.closePath();
    context.stroke();
    context.restore();

    const center = gridPoint(originX, originY, size, gridSize, { x: 5, y: 5 });
    context.save();
    context.strokeStyle = 'rgba(255,255,255,0.34)';
    context.strokeRect(center.x - cell * 0.32, center.y - cell * 0.32, cell * 0.64, cell * 0.64);
    context.restore();

    positions.forEach((position, index) => {
      const current = gridPoint(originX, originY, size, gridSize, position);
      const next = gridPoint(originX, originY, size, gridSize, position.next);
      const angle = Math.atan2(next.y - current.y, next.x - current.x) + Math.PI / 2;
      drawMouse(context, current.x, current.y, angle, cell * 0.88, index === 0 ? 1 : 0.7);
    });
  }, [pathA, pathB]);

  return <canvas ref={canvasRef} className="canvas-panel" aria-label="Asymmetric Social Representations in the Prefrontal Cortex for Cooperative Behavior" />;
}

function drawArrow(context, x1, y1, x2, y2, alpha) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  context.save();
  context.strokeStyle = `rgba(255,255,255,${alpha})`;
  context.fillStyle = `rgba(255,255,255,${alpha})`;
  context.lineWidth = 1.4;
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(x2 - 9 * Math.cos(angle - 0.45), y2 - 9 * Math.sin(angle - 0.45));
  context.lineTo(x2 - 9 * Math.cos(angle + 0.45), y2 - 9 * Math.sin(angle + 0.45));
  context.closePath();
  context.fill();
  context.restore();
}

function HMMCanvas() {
  const canvasRef = useCanvas((context, width, height, frame) => {
    clearCanvas(context, width, height);
    const paddingX = width * 0.1;
    const columns = 4;
    const stepWidth = (width - paddingX * 2) / (columns - 1);
    const yInput = height * 0.22;
    const yHidden = height * 0.5;
    const yObs = height * 0.78;
    const phase = Math.floor((frame / 38) % 50);
    const visibleSteps = Math.min(Math.floor(phase / 10) + 1, columns);
    const currentStep = Math.min(Math.floor(phase / 10), columns - 1);

    const getAlpha = (step, offset) => {
      if (phase >= step * 10 + offset) return step === currentStep ? 1 : 0.34;
      return 0.14;
    };

    const drawNode = (x, y, label, shape, alpha) => {
      context.save();
      context.strokeStyle = `rgba(255,255,255,${alpha})`;
      context.fillStyle = `rgba(255,255,255,${alpha})`;
      context.lineWidth = 1.7;
      if (shape === 'square') {
        context.strokeRect(x - 24, y - 24, 48, 48);
      } else {
        context.beginPath();
        context.arc(x, y, 26, 0, Math.PI * 2);
        context.stroke();
      }
      context.font = `14px ${MONO_FONT}`;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillText(label, x, y);
      context.restore();
    };

    for (let step = 0; step < visibleSteps; step += 1) {
      const x = paddingX + step * stepWidth;
      drawNode(x, yInput, `u_${step + 1}`, 'circle', getAlpha(step, 0));
      drawNode(x, yHidden, `z_${step + 1}`, 'circle', getAlpha(step, 3));
      drawNode(x, yObs, `x_${step + 1}`, 'square', getAlpha(step, 6));
      drawArrow(context, x, yInput + 30, x, yHidden - 30, getAlpha(step, 2));
      drawArrow(context, x, yHidden + 30, x, yObs - 30, getAlpha(step, 5));
      drawArrow(context, x - 8, yInput + 32, x - 16, yObs - 32, getAlpha(step, 7));
      if (step < visibleSteps - 1) {
        drawArrow(context, x + 31, yHidden, x + stepWidth - 31, yHidden, getAlpha(step + 1, 1));
      }
    }
  }, []);

  return <canvas ref={canvasRef} className="canvas-panel" aria-label="Bayesian Modeling Tutorial" />;
}

function SignalField() {
  const canvasRef = useCanvas((context, width, height, frame) => {
    clearCanvas(context, width, height);
    context.font = `11px ${MONO_FONT}`;
    context.fillStyle = 'rgba(255,255,255,0.7)';
    const glyphs = '01._:/\\\\|+-=*#';
    const cellWidth = 9;
    const cellHeight = 12;
    const cols = Math.ceil(width / cellWidth);
    const rows = Math.ceil(height / cellHeight);
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        const value = Math.sin((x + frame * 0.045) * 0.65) + Math.cos((y - frame * 0.035) * 0.8);
        const alpha = Math.max(0.12, Math.min(0.82, (value + 2) / 4));
        context.fillStyle = `rgba(255,255,255,${alpha})`;
        context.fillText(glyphs[(x * 7 + y * 11 + frame) % glyphs.length], x * cellWidth, y * cellHeight + 10);
      }
    }
  }, []);

  return <canvas ref={canvasRef} className="signal-canvas" aria-hidden="true" />;
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

function ProjectVisual({ type }) {
  if (type === 'hmm') return <HMMCanvas />;
  return <MouseSketchCanvas />;
}

function ProjectBlock({ project, index }) {
  return (
    <article className="project-block">
      <div className="project-output">
        <ProjectVisual type={project.type} />
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
  const projectList = useMemo(() => projects, []);

  return (
    <main className="site">
      <section className="hero" aria-label="Rudramani Singha">
        <div className="hero-grid">
          <div className="hero-terminal">
            <p className="boot-line">~/singha.io $ model --probabilistic --brain</p>
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
                  {contact.label}
                </a>
              ))}
            </nav>
          </div>
          <div className="hero-signal">
            <SignalField />
          </div>
        </div>
      </section>

      <section className="projects" aria-labelledby="selected-projects">
        <div className="section-heading">
          <h2 id="selected-projects">Selected Projects</h2>
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
