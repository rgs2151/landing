import Scene from './Scene';
import { researchPieces } from './pieces';
import { createLocalMotion } from './localMotion';
import ScenePortrait from './ScenePortrait';
import { tools } from '../shared/tools';
import { neuraljaxlibRevision } from '../content/neuraljaxlib';
import { biography } from '../content/site';
import { landingPreviews } from '../assets/landingPreviews';
import { NeuralJAXBrainGallery, neuraljaxAnimations } from '../shared/neuraljaxlib/animations';
import './styles.css';
import './mobile.css';

const placements = {
  pylba: { x: 970, y: 0, width: 175, angle: -5 },
  'neuraljax-cnn': { x: 785, y: 565, width: 126, angle: 10 },
  'neuraljax-rnn': { x: -20, y: 245, width: 180, angle: 9 },
  'neuraljax-autoencoder': { x: 1025, y: 175, width: 189.54, angle: 5 },
  'neuraljax-gan': { x: -40, y: 470, width: 226.3248, angle: -6 },
  'neuraljax-transformer': { x: 200, y: 610, width: 190, angle: 7 },
};
const scenePieces = [...researchPieces, ...tools.flatMap(tool => tool.figures.map(figure => ({
  id: figure.id, project: tool.id, kind: 'tool', Component: figure.Animation, ...placements[figure.id],
})))].map(piece => {
  if (piece.clip !== undefined) return piece;
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

export default function App() {
  return <Scene scenePieces={scenePieces} motion={createLocalMotion}
    Portrait={ScenePortrait} profileEntry projectArrival centeredStack sidebarCopyright aboutProfile={biography} tools={previewTools} navigationLabels={navigationLabels} />;
}
