import { Euler, Vector3 } from 'three';

// Motion reference: Thinking Orbs' independently oriented paths and depth cues.
// Our figures drift around stable anchors instead of completing revolutions.
export function createDriftMotion(index, x, y) {
  const track = index % 3 * 3 + Math.floor(index / 3);
  const phase = Math.atan2(y, x);
  const anchorX = (270 + track * 10) * 1.2 * Math.cos(phase);
  const anchorY = (160 + track * 6.25) * 1.2 * Math.sin(phase);
  const orientation = new Euler(.25 + index * .19, -.45 + index * .23, index * 2.399);
  const offset = index * 1.73;
  const direction = index % 2 ? -1 : 1;
  const keyframes = Array.from({ length: 181 }, (_, frame) => {
    const t = direction * frame / 180 * Math.PI * 2;
    const drift = new Vector3(
      54 * Math.sin(t + offset) + 18 * Math.sin(2 * t - offset),
      35 * Math.cos(2 * t + offset),
      78 * Math.sin(t - offset),
    ).applyEuler(orientation);
    const perspective = 1000 / (1000 - drift.z);
    const depth = Math.max(0, Math.min(1, (drift.z + 110) / 220));
    return {
      transform: `translate(${(anchorX + drift.x) * perspective - x}px, ${(anchorY + drift.y) * perspective - y}px) scale(${perspective})`,
      opacity: .74 + .26 * depth,
    };
  });
  return { keyframes, duration: 36000 + index * 2700 };
}
