import PrototypeB, { pieces } from '../b/PrototypeB';
import { createLocalMotion } from './localMotion';
import ScenePortrait from './ScenePortrait';
import './styles.css';

// Restore the pre-orbit spread and size balance; reuse all accepted assets/views.
const placements = {
  'nature-0': { x: 690, y: 65, width: 216 },
  'nature-1': { x: 170, y: 120 },
  'nature-2': { x: 150, y: 215 },
  'nature-3': { x: 960, y: 220 },
  autoregressive: { x: 365, y: 80 },
  signal: { x: 780, y: 515 },
  manifold: { x: 70, y: 325 },
  landscape: { x: 935, y: 355 },
  conversation: { x: 420, y: 480 },
};
const spread = pieces.map(piece => ({ ...piece, width: piece.width / 1.2, ...placements[piece.id] }));

export default function PrototypeC() {
  return <PrototypeB scenePieces={spread} motion={createLocalMotion} variant="c" Portrait={ScenePortrait} profileEntry />;
}
