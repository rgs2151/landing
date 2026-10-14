import PrototypeB from '../b/PrototypeB';
import { spread } from '../c/PrototypeC';
import { createLocalMotion } from '../c/localMotion';
import ScenePortrait from '../c/ScenePortrait';
import { tools } from '../../shared/tools';
import './styles.css';

const placements = {
  pylba: { x: 1010, y: 25, width: 175, angle: 7 },
  'neuraljax-cnn': { x: 850, y: 565, width: 140, angle: 10 },
  'neuraljax-rnn': { x: -20, y: 245, width: 180, angle: 9 },
  'neuraljax-autoencoder': { x: 985, y: 160, width: 234, angle: -12 },
  'neuraljax-gan': { x: 10, y: 610, width: 155, angle: -6 },
  'neuraljax-transformer': { x: 1040, y: 625, width: 190, angle: -7 },
};
const researchPlacements = {
  manifold: { x: 105, y: 295 },
  signal: { x: 835, y: 485 },
};
const scenePieces = [...spread.map(piece => ({ ...piece, ...researchPlacements[piece.id] })), ...tools.flatMap(tool => tool.figures.map(figure => ({
  id: figure.id, project: tool.id, kind: 'tool', Component: figure.Animation, ...placements[figure.id],
})))];
const navigationLabels = { research: 'Research Publications', packages: 'Tools & Packages' };

export default function PrototypeD() {
  return <PrototypeB scenePieces={scenePieces} motion={createLocalMotion} variant="d" className="prototype-c"
    Portrait={ScenePortrait} profileEntry tools={tools} navigationLabels={navigationLabels} />;
}
