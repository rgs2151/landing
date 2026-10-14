import PrototypeB, { pieces } from '../b/PrototypeB';
import { createLocalMotion } from './localMotion';

// Restore the pre-orbit spread and size balance; reuse all accepted assets/views.
const spread = pieces.map(piece => ({ ...piece, width: piece.width / 1.2 }));

export default function PrototypeC() {
  return <PrototypeB scenePieces={spread} motion={createLocalMotion} variant="c" />;
}
