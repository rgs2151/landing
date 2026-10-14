import PrototypeB from '../b/PrototypeB';
import { spread } from '../c/PrototypeC';
import { createLocalMotion } from '../c/localMotion';
import ScenePortrait from '../c/ScenePortrait';
import { tools } from '../../shared/tools';
import { neuraljaxlibRevision } from '../../content/neuraljaxlib';
import { biography } from '../../content/site';
import { landingPreviews } from '../../assets/landingPreviews';
import { NeuralJAXBrainGallery, neuraljaxAnimations } from '../../shared/neuraljaxlib/animations';
import './styles.css';

const placements = {
  pylba: { x: 970, y: 0, width: 175, angle: -5 },
  'neuraljax-cnn': { x: 785, y: 565, width: 126, angle: 10 },
  'neuraljax-rnn': { x: -20, y: 245, width: 180, angle: 9 },
  'neuraljax-autoencoder': { x: 1025, y: 175, width: 189.54, angle: 5 },
  'neuraljax-gan': { x: 0, y: 490, width: 174.096, angle: -6 },
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
  id: figure.id, project: tool.id, kind: 'tool', Component: figure.Animation, ...placements[figure.id],
})))].map(piece => {
  if (!piece.Component) return piece;
  const Component = landingPreviews[piece.id];
  if (!Component) throw new Error(`Export a landing preview before adding ${piece.id}`);
  return { ...piece, Component };
});
const navigationLabels = { research: 'Research Publications', packages: 'Tools & Packages' };
const previewTools = tools.map(tool => ({ ...tool,
  ...(tool.id === 'neuraljaxlib' ? {
    ...neuraljaxlibRevision, Gallery: NeuralJAXBrainGallery,
    abstractFigures: [
      { ...neuraljaxAnimations.find(asset => asset.key === 'cnn'), align: 'left', afterLines: 2 },
      { ...neuraljaxAnimations.find(asset => asset.key === 'gan'), align: 'right', afterLines: 5 },
    ],
  } : {}),
  navigationTitle: tool.id === 'pylba' ? 'pyLBA: Linear Ballistic Accumulators' : tool.title,
  year: tool.id === 'neuraljaxlib' ? 2026 : tool.year,
  navigationLogo: false,
})).sort((a, b) => b.year - a.year);

export default function PrototypeF() {
  return <PrototypeB scenePieces={scenePieces} motion={createLocalMotion} variant="f" className="prototype-c"
    Portrait={ScenePortrait} profileEntry projectArrival centeredStack sidebarCopyright aboutProfile={biography} tools={previewTools} navigationLabels={navigationLabels} />;
}
