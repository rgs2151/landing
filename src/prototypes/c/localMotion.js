export function createLocalMotion(index) {
  let seed = 713 + index * 101;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const keyframes = Array.from({ length: 17 }, (_, frame) => {
    const x = frame === 0 || frame === 16 ? 0 : (random() - .5) * 8;
    const y = frame === 0 || frame === 16 ? 0 : (random() - .5) * 6;
    return { transform: `translate(${x}px, ${y}px)`, easing: 'cubic-bezier(.45,0,.55,1)' };
  });
  return { keyframes, duration: 60000 + index * 3000 };
}
