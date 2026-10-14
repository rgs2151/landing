import { useEffect, useRef } from 'react';

export default function ProjectVideo({ src, poster, compact = false, active = true, ...props }) {
  const ref = useRef(null);
  useEffect(() => {
    const video = ref.current;
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    let visible = false;
    const sync = () => {
      if (active && visible && !document.hidden && !motion.matches) video.play().catch(() => {});
      else video.pause();
    };
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); });
    observer.observe(video);
    document.addEventListener('visibilitychange', sync);
    motion.addEventListener('change', sync);
    return () => {
      observer.disconnect(); video.pause();
      document.removeEventListener('visibilitychange', sync);
      motion.removeEventListener('change', sync);
    };
  }, [active, src]);
  return <video ref={ref} src={src} poster={poster} muted loop playsInline
    preload="none" controls={!compact} {...props} />;
}
