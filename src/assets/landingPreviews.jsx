import { useEffect, useRef } from 'react';
import neuraljaxlib from '../../public/assets/projects/neuraljaxlib/previews.json';
import neuralField from '../../public/assets/projects/neural-field-manifold/previews.json';
import steering from '../../public/assets/projects/robust-steering/previews.json';
import pylba from '../../public/assets/projects/pylba/previews.json';
import './landing-previews.css';

export const previewManifest = { ...neuraljaxlib, ...neuralField, ...steering, ...pylba };

export function LandingVideo({ clip, active = true, snapshotRef }) {
  const videoRef = useRef(null);
  useEffect(() => {
    const video = videoRef.current;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let visible = false, captured = false;
    const allowed = () => active && visible && !document.hidden && !reduced.matches && !captured;
    const sync = () => { if (allowed()) video.play().catch(() => {}); else video.pause(); };
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); });
    observer.observe(video);
    const end = () => { video.currentTime = clip.loopStart / 1000; sync(); };
    const ready = () => { video.dataset.ready = 'true'; sync(); };
    video.addEventListener('loadeddata', ready);
    video.addEventListener('ended', end);
    document.addEventListener('visibilitychange', sync);
    reduced.addEventListener('change', sync);
    if (video.readyState >= 2) ready();
    if (snapshotRef) snapshotRef.current = () => { captured = true; video.pause(); };
    return () => {
      observer.disconnect(); video.pause();
      video.removeEventListener('loadeddata', ready); video.removeEventListener('ended', end);
      document.removeEventListener('visibilitychange', sync); reduced.removeEventListener('change', sync);
      if (snapshotRef) snapshotRef.current = null;
    };
  }, [clip, active, snapshotRef]);
  return <div className="landing-video" data-preview={clip.id} style={{ aspectRatio: `${clip.layoutWidth} / ${clip.layoutHeight}` }}>
    <img src={`${clip.poster}?v=${clip.revision}`} alt="" aria-hidden="true" />
    <video ref={videoRef} src={`${clip.src}?v=${clip.revision}`} poster={`${clip.poster}?v=${clip.revision}`}
      muted playsInline loop={!clip.loopStart} preload="auto" aria-hidden="true" tabIndex={-1} />
  </div>;
}

export const landingPreviews = Object.fromEntries(Object.entries(previewManifest).map(([id, clip]) => [id, props => <LandingVideo {...props} clip={clip} />]));
