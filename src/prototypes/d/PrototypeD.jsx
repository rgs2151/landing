import PrototypeB from '../b/PrototypeB';
import { spread } from '../c/PrototypeC';
import { createLocalMotion } from '../c/localMotion';
import ScenePortrait from '../c/ScenePortrait';
import { tools } from '../../shared/tools';
import './styles.css';

const placements = {
  pylba: { x: 930, y: 0, width: 175, angle: 7 },
  'neuraljax-cnn': { x: 815, y: 565, width: 140, angle: 10 },
  'neuraljax-rnn': { x: -20, y: 245, width: 180, angle: 9 },
  'neuraljax-autoencoder': { x: 1025, y: 125, width: 234, angle: -7 },
  'neuraljax-gan': { x: 20, y: 470, width: 181.35, angle: -6 },
  'neuraljax-transformer': { x: 200, y: 600, width: 190, angle: 7 },
};
const researchPlacements = {
  manifold: { x: 105, y: 295 },
  signal: { x: 800, y: 485 },
};
const scenePieces = [...spread.map(piece => ({ ...piece, ...researchPlacements[piece.id] })), ...tools.flatMap(tool => tool.figures.map(figure => ({
  id: figure.id, project: tool.id, kind: 'tool', Component: figure.Animation, ...placements[figure.id],
})))];
const navigationLabels = { research: 'Research Publications', packages: 'Tools & Packages' };

export default function PrototypeD() {
  return <PrototypeB scenePieces={scenePieces} motion={createLocalMotion} variant="d" className="prototype-c"
    Portrait={ScenePortrait} profileEntry tools={tools} navigationLabels={navigationLabels} />;
}
