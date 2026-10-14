import PrototypeB from '../b/PrototypeB';
import { spread } from '../c/PrototypeC';
import { createLocalMotion } from '../c/localMotion';
import ScenePortrait from '../c/ScenePortrait';
import PyLBAAnimation from '../../shared/pylba/PyLBAAnimation';
import { pylba } from '../../content/pylba';
import './styles.css';

const tools = [{ ...pylba, Animation: PyLBAAnimation }];
const scenePieces = [...spread, {
  id: 'pylba', project: 'pylba', kind: 'tool', Component: PyLBAAnimation,
  x: 1010, y: 25, width: 175, angle: 7,
}];
const navigationLabels = { research: 'Research Publications', packages: 'Tools & Packages' };

export default function PrototypeD() {
  return <PrototypeB scenePieces={scenePieces} motion={createLocalMotion} variant="d" className="prototype-c"
    Portrait={ScenePortrait} profileEntry tools={tools} navigationLabels={navigationLabels} />;
}
