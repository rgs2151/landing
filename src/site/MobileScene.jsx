import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowLeft, Pause, Play, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { profile, paper } from '../content/site';
import { previewManifest } from '../assets/landingPreviews';
import NaturePreview, { NaturePreviewClock } from './NaturePreviews';

// A separate artboard keeps the accepted desktop placements completely intact.
const placements = {
  signal: [20, 20, 135, -6],
  pylba: [260, 8, 105, -5],
  'nature-1': [22, 116, 118, 9],
  autoregressive: [164, 130, 195, 5],
  'nature-2': [42, 235, 66, -9],
  'nature-3': [310, 246, 56, 8],
  'nature-0': [200, 428, 153, -8],
  manifold: [16, 438, 198, -6],
  landscape: [192, 626, 180, 5],
  'neuraljax-cnn': [14, 655, 120, 10],
  conversation: [12, 789, 228, -3],
  'neuraljax-autoencoder': [262, 798, 110, 5],
  'neuraljax-rnn': [12, 937, 155, 9],
  'neuraljax-gan': [224, 938, 147, -6],
  'neuraljax-transformer': [176, 1102, 190, 7],
};
const ignoreHover = () => {};

function MobilePiece({ piece, index, paused, replay, open, title }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [posterOnly, setPosterOnly] = useState(false);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      setVisible(entry.isIntersecting);
      if (entry.isIntersecting) setLoaded(true);
    }, { rootMargin: '100px' });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  const [x, y, width, angle] = placements[piece.id];
  const clip = previewManifest[piece.id];
  const poster = piece.clip === undefined ? previewManifest[piece.id].poster : paper.clips[piece.clip].poster;
  function checkTransparency(event) {
    const video = event.target;
    if (video.tagName !== 'VIDEO' || !video.videoWidth) return;
    // Some phone decoders play VP9 but discard alpha. Keep the clean poster there.
    const sample = document.createElement('canvas');
    sample.width = 2; sample.height = 2;
    const ctx = sample.getContext('2d', { willReadFrequently: true });
    for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
      ctx.drawImage(video, x * (video.videoWidth - 1), y * (video.videoHeight - 1), 1, 1, x, y, 1, 1);
    }
    const pixels = ctx.getImageData(0, 0, 2, 2).data;
    if ([3, 7, 11, 15].every(index => pixels[index] === 255)) setPosterOnly(true);
  }
  return <button ref={ref} className="pm-piece" data-piece={piece.id} data-kind={piece.kind}
    data-visible={visible} style={{ left: x, top: y, width, aspectRatio: clip ? `${clip.layoutWidth} / ${clip.layoutHeight}` : '1', '--pm-angle': `${angle}deg`, '--pm-delay': `${-index * 1.7}s` }}
    aria-label={`${title}: ${piece.id}`} onClick={event => open(piece.project, event.currentTarget)} onLoadedDataCapture={checkTransparency}>
    <div className="pm-piece-motion" key={replay}>
      {loaded && !posterOnly ? piece.Component ? <piece.Component compact active={!paused && visible} />
        : <NaturePreview clip={piece.clip} poster={poster} /> : <img src={poster} alt="" loading="lazy" />}
    </div>
  </button>;
}

export default function MobileScene({ selected, open, close, scenePieces, tools, aboutProfile, labels,
  Navigation, ProjectDetail, paused, setPaused, replay, setReplay, reduced }) {
  const host = useRef(null);
  const heading = useRef(null);
  const reading = useRef(null);
  const savedScroll = useRef(0);
  const previous = useRef(selected);
  const [scale, setScale] = useState(1);
  const [zoom, setZoom] = useState(false);
  const hasWideFigure = selected && selected !== 'about' && !tools.some(tool => tool.id === selected && !tool.Gallery);
  useLayoutEffect(() => {
    if (selected) return;
    const resize = new ResizeObserver(([entry]) => setScale(entry.contentRect.width / 390));
    resize.observe(host.current);
    return () => resize.disconnect();
  }, [selected]);
  useLayoutEffect(() => {
    setZoom(false);
    if (selected) {
      window.scrollTo({ top: 0, behavior: 'instant' });
      heading.current?.focus({ preventScroll: true });
      reading.current?.querySelectorAll('.pb-project-media,.pb-tool-gallery').forEach(node => {
        node.tabIndex = 0;
        node.setAttribute('aria-label', 'Project animation');
      });
    } else if (previous.current) {
      window.scrollTo({ top: savedScroll.current, behavior: 'instant' });
      document.querySelector('.pm-name button')?.focus({ preventScroll: true });
    }
    previous.current = selected;
  }, [selected]);
  function enter(id, target) {
    if (!selected) savedScroll.current = window.scrollY;
    open(id, target);
  }
  return <main className="site pm-mobile" data-selected={selected || ''} data-paused={paused || reduced} data-zoom={zoom}>
    <header className="pm-header">
      <Navigation selected={selected} open={enter} hover={ignoreHover} tools={tools} labels={labels} omitAccepted touch />
    </header>
    {selected ? <>
      <div className="pm-toolbar">
        <button onClick={close} aria-label="Return to composition" title="Return to composition"><ArrowLeft size={22} />
          {selected !== 'about' && <span className="pm-stack" aria-hidden="true"><img src={previewManifest.manifold.poster} alt="" /><img src={profile.photo} alt="" /></span>}
        </button>
        {hasWideFigure && <button onClick={() => setZoom(value => !value)} aria-pressed={zoom}
          aria-label={zoom ? 'Fit figures to screen' : 'Enlarge figures'} title={zoom ? 'Fit figures to screen' : 'Enlarge figures'}>
          {zoom ? <ZoomOut size={21} /> : <ZoomIn size={21} />}
        </button>}
      </div>
      <section className="pm-reading" ref={reading} key={selected}>
        <ProjectDetail id={selected} headingRef={heading} portraitFirst portraitPhoto tools={tools} omitAccepted aboutProfile={aboutProfile} />
      </section>
    </> : <section className="pm-composition" aria-label="Interactive research composition" ref={host} style={{ height: 1360 * scale }}>
      <div className="pm-artboard" style={{ transform: `scale(${scale})` }}>
        <h1 className="pm-name"><button onClick={event => enter('about', event.currentTarget)} aria-label={`About ${profile.name}`}>
          <span>{profile.name.split(' ')[0]}</span><span>{profile.name.split(' ').slice(1).join(' ')}</span>
        </button></h1>
        <button className="pm-portrait" onClick={event => enter('about', event.currentTarget)} aria-label={`About ${profile.name}`}>
          <img src={profile.photo} alt={profile.name} />
        </button>
        <NaturePreviewClock active={!paused && !reduced} replay={replay}>
          {scenePieces.map((piece, index) => <MobilePiece key={piece.id} piece={piece} index={index} paused={paused || reduced}
            replay={replay} open={enter} title={tools.find(tool => tool.id === piece.project)?.shortTitle || piece.project.replaceAll('-', ' ')} />)}
        </NaturePreviewClock>
      </div>
    </section>}
    <footer className="pm-footer"><span>&copy; 2026 {profile.name}</span>
      {!selected && !reduced && <div className="pm-playback">
        <button onClick={() => setPaused(value => !value)} aria-label={paused ? 'Play composition' : 'Pause composition'} title={paused ? 'Play' : 'Pause'}>{paused ? <Play size={17} /> : <Pause size={17} />}</button>
        <button onClick={() => setReplay(value => value + 1)} aria-label="Replay composition" title="Replay"><RotateCcw size={17} /></button>
      </div>}
    </footer>
  </main>;
}
