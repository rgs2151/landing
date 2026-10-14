import PrototypeB, { pieces } from '../b/PrototypeB';
import { createLocalMotion } from './localMotion';
import ScenePortrait from './ScenePortrait';
import './styles.css';

// Restore the pre-orbit spread and size balance; reuse all accepted assets/views.
const placements = {
  'nature-0': { x: 720, y: -10, width: 216 },
  'nature-1': { x: 150, y: 15 },
  'nature-2': { x: 65, y: 210 },
  'nature-3': { x: 1050, y: 125 },
  autoregressive: { x: 365, y: 50 },
  signal: { x: 785, y: 575 },
  manifold: { x: 0, y: 355 },
  landscape: { x: 940, y: 445 },
  conversation: { x: 400, y: 510 },
};
const spread = pieces.map(piece => ({ ...piece, width: piece.width / 1.2, ...placements[piece.id] }));

export default function PrototypeC() {
  return <PrototypeB scenePieces={spread} motion={createLocalMotion} variant="c" Portrait={ScenePortrait} />;
}
