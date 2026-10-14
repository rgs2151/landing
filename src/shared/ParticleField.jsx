import { useEffect, useRef } from 'react';

// A visual sketch of a representation field, not a scientific model.
export default function ParticleField({ steering = 0, variant = 'sphere', className = '' }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const context = canvas.getContext('2d');
    const draw = () => {
      const { width, height } = canvas.getBoundingClientRect();
      const dpr = Math.min(devicePixelRatio, 2);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.scale(dpr, dpr);
      const radius = Math.min(width, height) * 0.39;
      const points = [];
      for (let i = 0; i < 1100; i++) {
        const y = 1 - (i / 1099) * 2;
        const ring = Math.sqrt(1 - y * y);
        const angle = i * Math.PI * (3 - Math.sqrt(5));
        let x = Math.cos(angle) * ring;
        let z = Math.sin(angle) * ring;
        if (variant === 'folded') {
          const fold = 1 + 0.1 * Math.sin(angle * 5 + y * 12);
          x *= fold * 1.15;
          z *= fold;
        }
        x += steering * 0.25 * Math.sin(y * 4 + z * 2);
        const rotation = 0.4 + steering * 0.6;
        const rx = x * Math.cos(rotation) - z * Math.sin(rotation);
        const rz = x * Math.sin(rotation) + z * Math.cos(rotation);
        points.push({ x: rx, y, z: rz });
      }
      points.sort((a, b) => a.z - b.z);
      for (const point of points) {
        context.fillStyle = `rgba(239,239,233,${0.18 + (point.z + 1) * 0.35})`;
        context.beginPath();
        context.arc(width / 2 + point.x * radius, height / 2 + point.y * radius * 0.91,
          0.7 + (point.z + 1) * 0.6, 0, Math.PI * 2);
        context.fill();
      }
    };
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    draw();
    return () => observer.disconnect();
  }, [steering, variant]);

  return <canvas className={className} ref={ref} role="img" aria-label="Abstract dotted representation field" />;
}
