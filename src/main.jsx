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

const MARLAX_GRID_SIZE = 11;
const MARLAX_FRAME_MS = 140;
// Encoded from scripts/artifacts/logs.parquet for scripts/animate.ipynb, regime 1, frames 0-500.
const MARLAX_POSITION_STREAM = [
  '20323033403450356045705560456146624763486449654a554a5649574a584a595a5a5a350645055504561457155825592659275a286a295a395949',
  '594a5a5a8865985588567857685869596a5a5a5aa455a354935383528251725162506151605150505614550454145324522351335032513150414040',
  '50503125313530452055305440535052505150501365125513451435042505150505819a828a837a846a745a755a655a554a45493548254715461536',
  '0535052515150505439944984597558765877586859695a6a5a5646a655a554a565a576a586959595a5a468245825583547353725262516150605050',
  '533054205510561157215831484149424a433a443a454a465a474a484949594a5a5a55956596758685969595a5a52598359845975587658775868596',
  '95a6a5a564a965a855a765a775a685969595a5a5a375a265a155a265a375a485a495a5a54a455a555a565a57595859595a5aaa45aa55a965a875a785',
  'a695a5a5214a314941485147504660457055715461535152505150505796569755874577356725571547154614360426141604060505615360547055',
  '6054605350525051505096759765985588567857685869596a5a5a5aaa3a9a4a8a497a487a477a466a455a555a564a5749584a595a5a777076717572',
  '6573558365847585859595a5a5a5769366936593558356735774587559765977497859795a7a4a6a5a5a93199419841a742a752a653a554a45493548',
  '254715461536053505251515050564846584558356735774587559765977497859795a7a4a6a5a5a671a662a653a554a565a576a586959595a5a9a32',
  '8a428a437a446a455a555a564a5749584a595a5a7837684769466a455a556217521853185428553856485758585959595a5a07730774077507651755',
  '27563757385848594949594a5a5a53135414550445053515250515060505622263235313541455045403531352235133503251315141505150502559',
  '3559455a554a654975598569956895679577a5879586949695a6a5a5450955194518351725161515050527a937a84798469845975587658675968596',
  '9595a5a5476548555856595749584a595a5a69336943595359545a555a565a57595859595a5aa5919591859275826582558365847585859595a5a5a5',
  '68946795669665975587568857785868585859595a5a4146404530553154415351525051505065a155a2569246824772377338633964496559665967',
  '5868585859595a5a355536563757385848594949594a5a5aaa409a418a428a437a446a455a555956595749584a595a5a539754975587568857785868',
  '585859595a5a6a565a557a756a655a555a565a57595859595a5a92959185907580657055716572758275'
].join('');
const MARLAX_REWARD_STREAM = [
  '000001111111111111002222222222220333333333333333330444444444400033333301111110000000555555555500044444400333333006666666',
  '002222222222222222222220004444440022222200222222022222205555550000004444440066666666666003333330033333330000000111111000',
  '022222200011111111111000000555555555500111111111110003333330000011111100006000022222200002222222200111111000033333333330',
  '004444444444440555555000005555550333333000022222200000444444000011111110044444401111111111111122222222000000333333001111',
  '111060022222200005555'
].join('');
const MARLAX_REWARD_TOKENS = ['', 'ul', 'ur', 'ud', 'rd', 'rl', 'dl'];
const MARLAX_COLLECTED_FRAMES = new Set([
  17, 31, 39, 49, 60, 69, 76, 93, 102, 110, 119, 136, 142, 151, 159, 167, 174, 181, 193, 206, 214, 223, 236, 246, 260, 276, 289,
  298, 309, 324, 336, 344, 358, 373, 380, 391, 398, 408, 419, 430, 438, 453, 461, 473, 482, 492
]);

function calculateNotebookAngle(xNew, yNew, xOld, yOld) {
  const dx = xNew - xOld;
  const dy = yNew - yOld;
  if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return null;
  return (Math.atan2(dy, dx) * 180) / Math.PI - 90;
}

function smoothNotebookAngle(currentAngle, nextAngle) {
  let angleDiff = nextAngle - currentAngle;
  while (angleDiff > 180) angleDiff -= 360;
  while (angleDiff < -180) angleDiff += 360;
  return currentAngle + angleDiff * 0.8;
}

function decodeMarlaxFrames() {
  const frames = [];
  const agentAngles = [0, 0];
  const previous = [
    { x: Number.parseInt(MARLAX_POSITION_STREAM[0], 36), y: Number.parseInt(MARLAX_POSITION_STREAM[1], 36) },
    { x: Number.parseInt(MARLAX_POSITION_STREAM[2], 36), y: Number.parseInt(MARLAX_POSITION_STREAM[3], 36) }
  ];

  for (let index = 0; index < MARLAX_REWARD_STREAM.length; index += 1) {
    const offset = index * 4;
    const agents = [
      { x: Number.parseInt(MARLAX_POSITION_STREAM[offset], 36), y: Number.parseInt(MARLAX_POSITION_STREAM[offset + 1], 36) },
      { x: Number.parseInt(MARLAX_POSITION_STREAM[offset + 2], 36), y: Number.parseInt(MARLAX_POSITION_STREAM[offset + 3], 36) }
    ];

    agents.forEach((agent, agentIndex) => {
      const nextAngle = calculateNotebookAngle(agent.x, agent.y, previous[agentIndex].x, previous[agentIndex].y);
      if (nextAngle !== null) agentAngles[agentIndex] = smoothNotebookAngle(agentAngles[agentIndex], nextAngle);
      previous[agentIndex] = agent;
    });

    frames.push({
      agents: agents.map((agent, agentIndex) => ({ ...agent, angle: agentAngles[agentIndex] })),
      reward: MARLAX_REWARD_TOKENS[Number.parseInt(MARLAX_REWARD_STREAM[index], 36)],
      collected: MARLAX_COLLECTED_FRAMES.has(index)
    });
  }

  return frames;
}

const MARLAX_FRAMES = decodeMarlaxFrames();

function useCanvas(draw, deps = []) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const startTime = performance.now();
    let frame = 0;
    let animationId = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(context, rect.width, rect.height, frame, performance.now() - startTime);
    };

    const render = () => {
      const rect = canvas.getBoundingClientRect();
      draw(context, rect.width, rect.height, frame, performance.now() - startTime);
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

function drawMouse(context, x, y, angleDegrees, scale, alpha = 1) {
  context.save();
  context.translate(x, y);
  context.rotate((-angleDegrees * Math.PI) / 180);
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

function getNotebookRewardCoord(token) {
  if (token === 'u') return { x: 5, y: 10 };
  if (token === 'd') return { x: 5, y: 0 };
  if (token === 'l') return { x: 0, y: 5 };
  if (token === 'r') return { x: 10, y: 5 };
  return null;
}

function getNotebookRewardCoords(reward) {
  if (!reward) return [];
  return reward
    .split('')
    .map((token) => getNotebookRewardCoord(token))
    .filter(Boolean);
}

function drawRewardMarker(context, point, cell) {
  context.save();
  context.strokeStyle = 'rgba(255,255,255,0.82)';
  context.lineWidth = 1.6;
  context.beginPath();
  context.moveTo(point.x, point.y - cell * 0.35);
  context.lineTo(point.x + cell * 0.35, point.y);
  context.lineTo(point.x, point.y + cell * 0.35);
  context.lineTo(point.x - cell * 0.35, point.y);
  context.closePath();
  context.stroke();
  context.restore();
}

function MouseSketchCanvas() {
  const canvasRef = useCanvas((context, width, height, frame, elapsedMs) => {
    clearCanvas(context, width, height);
    const size = Math.min(width, height) * 0.78;
    const originX = (width - size) / 2;
    const originY = (height - size) / 2;
    const gridSize = MARLAX_GRID_SIZE;
    drawGrid(context, originX, originY, size, gridSize - 1);
    const cell = size / (gridSize - 1);
    const stepIndex = Math.floor(elapsedMs / MARLAX_FRAME_MS) % MARLAX_FRAMES.length;
    const step = MARLAX_FRAMES[stepIndex];

    const rewardCoords = getNotebookRewardCoords(step.reward);
    rewardCoords.forEach((reward) => {
      drawRewardMarker(context, gridPoint(originX, originY, size, gridSize, reward), cell);
    });

    const center = gridPoint(originX, originY, size, gridSize, { x: 5, y: 5 });
    context.save();
    context.strokeStyle = 'rgba(255,255,255,0.34)';
    context.strokeRect(center.x - cell * 0.32, center.y - cell * 0.32, cell * 0.64, cell * 0.64);
    context.restore();

    step.agents.forEach((agent, index) => {
      const point = gridPoint(originX, originY, size, gridSize, agent);
      drawMouse(context, point.x, point.y, agent.angle, cell, index === 0 ? 1 : 0.72);
    });
  }, []);

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

function drawCurvedArrow(context, x1, y1, controlX, controlY, x2, y2, alpha) {
  const angle = Math.atan2(y2 - controlY, x2 - controlX);
  context.save();
  context.strokeStyle = `rgba(255,255,255,${alpha})`;
  context.fillStyle = `rgba(255,255,255,${alpha})`;
  context.lineWidth = 1.4;
  context.beginPath();
  context.moveTo(x1, y1);
  context.quadraticCurveTo(controlX, controlY, x2, y2);
  context.stroke();
  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(x2 - 9 * Math.cos(angle - 0.45), y2 - 9 * Math.sin(angle - 0.45));
  context.lineTo(x2 - 9 * Math.cos(angle + 0.45), y2 - 9 * Math.sin(angle + 0.45));
  context.closePath();
  context.fill();
  context.restore();
}

function drawRoundedRect(context, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + safeRadius, y);
  context.lineTo(x + width - safeRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + safeRadius);
  context.lineTo(x + width, y + height - safeRadius);
  context.quadraticCurveTo(x + width, y + height, x + width - safeRadius, y + height);
  context.lineTo(x + safeRadius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - safeRadius);
  context.lineTo(x, y + safeRadius);
  context.quadraticCurveTo(x, y, x + safeRadius, y);
  context.closePath();
}

function drawMathLabel(context, x, y, base, subscript, alpha) {
  const font = 'Georgia, Times New Roman, serif';
  const baseSize = 18;
  const subscriptSize = 11;
  context.save();
  context.fillStyle = `rgba(255,255,255,${alpha})`;
  context.textBaseline = 'alphabetic';
  context.font = `italic ${baseSize}px ${font}`;
  const baseWidth = context.measureText(base).width;
  context.font = `${subscriptSize}px ${font}`;
  const subscriptWidth = context.measureText(subscript).width;
  const startX = x - (baseWidth + subscriptWidth) / 2;
  context.font = `italic ${baseSize}px ${font}`;
  context.fillText(base, startX, y + baseSize * 0.34);
  context.font = `${subscriptSize}px ${font}`;
  context.fillText(subscript, startX + baseWidth + 1, y + baseSize * 0.55);
  context.restore();
}

function HMMCanvas() {
  const canvasRef = useCanvas((context, width, height, frame, elapsedMs) => {
    clearCanvas(context, width, height);
    const bounds = { xMin: 0.8, xMax: 10.2, yMin: -0.1, yMax: 5.7 };
    const padding = 20;
    const scale = Math.min((width - padding * 2) / (bounds.xMax - bounds.xMin), (height - padding * 2) / (bounds.yMax - bounds.yMin));
    const plotWidth = (bounds.xMax - bounds.xMin) * scale;
    const plotHeight = (bounds.yMax - bounds.yMin) * scale;
    const originX = (width - plotWidth) / 2;
    const originY = (height - plotHeight) / 2;
    const columns = 4;
    const inputY = 5;
    const hiddenY = 3;
    const obsY = 0.6;
    const phase = Math.floor((elapsedMs / 100) % 50);
    const visibleSteps = Math.min(Math.floor(phase / 10) + 1, columns);
    const currentStep = Math.floor(phase / 10);

    const mapPoint = (x, y) => ({
      x: originX + (x - bounds.xMin) * scale,
      y: originY + (bounds.yMax - y) * scale
    });

    const getAlpha = (step, offset) => {
      if (phase >= step * 10 + offset) return step === currentStep ? 1 : 0.4;
      return 0.3;
    };

    const drawNode = (x, y, base, subscript, shape, alpha) => {
      const point = mapPoint(x, y);
      context.save();
      context.strokeStyle = `rgba(255,255,255,${alpha})`;
      context.fillStyle = '#050505';
      context.lineWidth = 1.7;
      if (shape === 'square') {
        const half = 0.5 * scale;
        drawRoundedRect(context, point.x - half, point.y - half, half * 2, half * 2, 0.1 * scale);
        context.fill();
        context.stroke();
      } else {
        context.beginPath();
        context.arc(point.x, point.y, 0.6 * scale, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      }
      context.restore();
      drawMathLabel(context, point.x, point.y, base, subscript, alpha);
    };

    for (let step = 0; step < visibleSteps; step += 1) {
      const x = 2 + step * 2.5;
      const inputStart = mapPoint(x, inputY - 0.6);
      const hiddenEnd = mapPoint(x, hiddenY + 0.6);
      const hiddenStart = mapPoint(x, hiddenY - 0.6);
      const obsEnd = mapPoint(x, obsY + 0.6);
      drawArrow(context, inputStart.x, inputStart.y, hiddenEnd.x, hiddenEnd.y, getAlpha(step, 2));
      drawArrow(context, hiddenStart.x, hiddenStart.y, obsEnd.x, obsEnd.y, getAlpha(step, 5));

      const curveStart = mapPoint(x - 0.2, inputY - 0.6);
      const curveEnd = mapPoint(x - 0.4, obsY + 0.6);
      const curveControl = mapPoint(x - 1.1, (inputY + obsY) / 2);
      drawCurvedArrow(context, curveStart.x, curveStart.y, curveControl.x, curveControl.y, curveEnd.x, curveEnd.y, getAlpha(step, 7));
    }

    for (let step = 0; step < visibleSteps - 1; step += 1) {
      const x1 = 2 + step * 2.5;
      const x2 = 2 + (step + 1) * 2.5;
      const start = mapPoint(x1 + 0.6, hiddenY);
      const end = mapPoint(x2 - 0.6, hiddenY);
      const alpha = phase >= (step + 1) * 10 + 1 ? (step === currentStep || step + 1 === currentStep ? 1 : 0.4) : 0.3;
      drawArrow(context, start.x, start.y, end.x, end.y, alpha);
    }

    for (let step = 0; step < visibleSteps; step += 1) {
      const x = 2 + step * 2.5;
      const subscript = String(step + 1);
      drawNode(x, inputY, 'u', subscript, 'circle', getAlpha(step, 0));
      drawNode(x, hiddenY, 'z', subscript, 'circle', getAlpha(step, 3));
      drawNode(x, obsY, 'x', subscript, 'square', getAlpha(step, 6));
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
