import { useEffect, useRef } from 'react';
import neuraljax from '../../public/assets/projects/neuraljaxlib/mobile-previews.json';
import neuralField from '../../public/assets/projects/neural-field-manifold/mobile-previews.json';
import steering from '../../public/assets/projects/robust-steering/mobile-previews.json';
import pylba from '../../public/assets/projects/pylba/mobile-previews.json';

export const mobilePreviews = { ...neuraljax, ...neuralField, ...steering, ...pylba };

// Decode an ordinary H.264 movie and restore its separately packed alpha plane.
// This is cached video playback, not a second implementation of the animations.
export default function MobileVideo({ clip, active }) {
  const host = useRef(null), videoRef = useRef(null), canvasRef = useRef(null);
  const encoded = mobilePreviews[clip.id];
  useEffect(() => {
    const video = videoRef.current, canvas = canvasRef.current;
    const { width, height } = encoded;
    const packed = document.createElement('canvas');
    packed.width = width * 2; packed.height = height;
    canvas.width = width; canvas.height = height;
    const input = packed.getContext('2d', { willReadFrequently: true });
    const output = canvas.getContext('2d');
    const frame = output.createImageData(width, height);
    let visible = false, callback = 0, raf = 0, lastTime = -1, lastPaint = -Infinity;
    const allowed = () => active && visible && !document.hidden;
    function draw() {
      if (video.readyState < 2 || video.currentTime === lastTime) return;
      input.drawImage(video, 0, 0, width * 2, height);
      const data = input.getImageData(0, 0, width * 2, height).data;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const source = (y * width * 2 + x) * 4, target = (y * width + x) * 4;
        frame.data[target] = data[source];
        frame.data[target + 1] = data[source + 1];
        frame.data[target + 2] = data[source + 2];
        const alpha = data[source + width * 4];
        frame.data[target + 3] = alpha < 3 ? 0 : alpha > 252 ? 255 : alpha;
      }
      output.putImageData(frame, 0, 0);
      canvas.dataset.ready = 'true';
      canvas.dataset.time = String(video.currentTime);
      lastTime = video.currentTime;
    }
    function stop() {
      if (callback) video.cancelVideoFrameCallback(callback);
      cancelAnimationFrame(raf); callback = 0; raf = 0;
    }
    function tick(now) {
      callback = 0; raf = 0;
      if (!allowed() || video.paused) return;
      if (now - lastPaint >= 1000 / 15 - 1) { draw(); lastPaint = now; }
      schedule();
    }
    function schedule() {
      if (callback || raf || !allowed() || video.paused) return;
      if (video.requestVideoFrameCallback) callback = video.requestVideoFrameCallback(tick);
      else raf = requestAnimationFrame(tick);
    }
    function sync() {
      if (allowed()) video.play().then(schedule).catch(() => {});
      else { video.pause(); stop(); }
    }
    function ready() { draw(); sync(); }
    function loop() { video.currentTime = clip.loopStart / 1000; sync(); }
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); });
    observer.observe(host.current);
    video.addEventListener('loadeddata', ready);
    video.addEventListener('playing', schedule);
    video.addEventListener('pause', stop);
    video.addEventListener('ended', loop);
    video.addEventListener('seeked', draw);
    document.addEventListener('visibilitychange', sync);
    if (video.readyState >= 2) ready();
    return () => {
      stop(); observer.disconnect(); video.pause();
      video.removeEventListener('loadeddata', ready);
      video.removeEventListener('playing', schedule);
      video.removeEventListener('pause', stop);
      video.removeEventListener('ended', loop);
      video.removeEventListener('seeked', draw);
      document.removeEventListener('visibilitychange', sync);
    };
  }, [clip, encoded, active]);
  return <div className="pm-packed-video" data-preview={clip.id} ref={host}
    style={{ aspectRatio: `${clip.layoutWidth} / ${clip.layoutHeight}` }}>
    <img src={`${clip.poster}?v=${clip.revision}`} alt="" />
    <video ref={videoRef} data-packed="true" src={`${encoded.src}?v=${encoded.revision}`}
      muted playsInline preload="auto" loop={!clip.loopStart} aria-hidden="true" tabIndex={-1} />
    <canvas ref={canvasRef} aria-hidden="true" />
  </div>;
}
