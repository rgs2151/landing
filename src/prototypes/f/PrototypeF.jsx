import PrototypeB from '../b/PrototypeB';
import { spread } from '../c/PrototypeC';
import { createLocalMotion } from '../c/localMotion';
import ScenePortrait from '../c/ScenePortrait';
import { tools } from '../../shared/tools';
import { neuraljaxlibRevision } from '../../content/neuraljaxlib';
import { biography } from '../../content/site';
import { cachedNeuraljaxPreviews } from '../../shared/neuraljaxlib/CachedPreviews';
import './styles.css';

const placements = {
  pylba: { x: 970, y: 0, width: 175, angle: -5 },
  'neuraljax-cnn': { x: 785, y: 565, width: 126, angle: 10 },
  'neuraljax-rnn': { x: -20, y: 245, width: 180, angle: 9 },
  'neuraljax-autoencoder': { x: 1025, y: 175, width: 189.54, angle: 5 },
  'neuraljax-gan': { x: 30, y: 460, width: 145.08, angle: -6 },
  'neuraljax-transformer': { x: 200, y: 610, width: 190, angle: 7 },
};
const researchPlacements = {
  'nature-0': { y: 35, width: 280.8 },
  autoregressive: { x: 345, y: 120 },
  conversation: { x: 420, y: 470 },
  manifold: { x: 105, y: 295 },
  signal: { x: 790, y: 485 },
};
const scenePieces = [...spread.map(piece => ({ ...piece, ...researchPlacements[piece.id] })), ...tools.flatMap(tool => tool.figures.map(figure => ({
  id: figure.id, project: tool.id, kind: 'tool', Component: cachedNeuraljaxPreviews[figure.id] || figure.Animation, ...placements[figure.id],
})))];
const navigationLabels = { research: 'Research Publications', packages: 'Tools & Packages' };
const previewTools = tools.map(tool => ({ ...tool,
  ...(tool.id === 'neuraljaxlib' ? neuraljaxlibRevision : {}),
  navigationTitle: tool.id === 'pylba' ? 'pyLBA: Linear Ballistic Accumulators' : tool.title,
  year: tool.id === 'neuraljaxlib' ? 2026 : tool.year,
  navigationLogo: false,
})).sort((a, b) => b.year - a.year);

export default function PrototypeF() {
  return <PrototypeB scenePieces={scenePieces} motion={createLocalMotion} variant="f" className="prototype-c"
    Portrait={ScenePortrait} profileEntry projectArrival centeredStack sidebarCopyright aboutProfile={biography} tools={previewTools} navigationLabels={navigationLabels} />;
}
