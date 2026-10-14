import PrototypeB, { pieces } from '../b/PrototypeB';
import { createLocalMotion } from './localMotion';
import ScenePortrait from './ScenePortrait';
import './styles.css';

// Restore the pre-orbit spread and size balance; reuse all accepted assets/views.
const placements = {
  'nature-0': { x: 755, y: 105, width: 216 },
  'nature-1': { x: 205, y: -15 },
  autoregressive: { x: 505, y: 100 },
  signal: { x: 785, y: 525 },
};
const spread = pieces.map(piece => ({ ...piece, width: piece.width / 1.2, ...placements[piece.id] }));

export default function PrototypeC() {
  return <PrototypeB scenePieces={spread} motion={createLocalMotion} variant="c" Portrait={ScenePortrait} />;
}
