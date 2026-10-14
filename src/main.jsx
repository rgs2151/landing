import { StrictMode, useEffect, useRef, useState } from 'react';
import { Moon, Sun, Pause, Play } from 'lucide-react';
import { createRoot } from 'react-dom/client';
import { profile, paper } from './content/site';

function PaperVideo() {
  const videoRef = useRef(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    const mobile = matchMedia('(max-width: 740px)');
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    let nearby = false;
    let visible = false;
    let pausedByUser = reducedMotion.matches;
    const syncPlayback = () => {
      if (visible && !document.hidden && !pausedByUser) video.play().catch(() => {});
      else video.pause();
    };
    const setSource = () => {
      const layout = mobile.matches ? 'grid' : 'wide';
      video.poster = `/assets/videos/paper-${layout}.jpg`;
      if (nearby) {
        video.src = `/assets/videos/paper-${layout}.mp4`;
        video.load();
        syncPlayback();
      }
    };
    const loader = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !nearby) {
        nearby = true;
        setSource();
      }
    }, { rootMargin: '200px' });
    const playback = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      syncPlayback();
    });
    const onToggle = () => {
      pausedByUser = !video.paused;
      if (pausedByUser) video.pause();
      else video.play().catch(() => {});
    };
    video.addEventListener('toggle-playback', onToggle);
    document.addEventListener('visibilitychange', syncPlayback);
    mobile.addEventListener('change', setSource);
    setSource();
    loader.observe(video);
    playback.observe(video);
    return () => {
      loader.disconnect();
      playback.disconnect();
      mobile.removeEventListener('change', setSource);
      document.removeEventListener('visibilitychange', syncPlayback);
      video.removeEventListener('toggle-playback', onToggle);
      video.pause();
      video.removeAttribute('src');
      video.load();
    };
  }, []);

  return (
    <div className="paper-video">
      <video ref={videoRef} muted loop playsInline preload="none"
        aria-label="Four panels of cooperative behavior from the paper's supplementary videos"
        onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
      <button type="button" className="video-toggle"
        aria-label={playing ? 'Pause video' : 'Play video'}
        title={playing ? 'Pause video' : 'Play video'}
        onClick={() => videoRef.current.dispatchEvent(new Event('toggle-playback'))}>
        {playing ? <Pause size={18} aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
      </button>
    </div>
  );
}

function LandingPage() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || 'dark');

  function toggleTheme() {
    const nextTheme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = nextTheme;
    setTheme(nextTheme);
    try {
      localStorage.setItem('theme', nextTheme);
    } catch {
      // Theme switching still works when browser storage is unavailable.
    }
  }

  return (
    <table className="page-shell">
      <tbody>
        <tr className="row-reset">
          <td className="cell-reset">
            <div className="theme-toolbar">
              <button
                className="theme-toggle"
                type="button"
                onClick={toggleTheme}
                aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              >
                {theme === 'dark' ? <Sun size={20} aria-hidden="true" /> : <Moon size={20} aria-hidden="true" />}
              </button>
            </div>
            <table className="content-table">
              <tbody>
                <tr className="row-reset intro-row">
                  <td className="intro-text">
                    <p className="text-center">
                      <span className="site-name">{profile.name}</span>
                    </p>
                    <p>
                      {profile.role} <a href={profile.institutionUrl} target="_blank" rel="noopener noreferrer">{profile.institution}</a>{profile.affiliation}
                      {profile.statement}
                    </p>
                    <p className="contact-block text-center">
                      <span>{profile.email}</span>
                      <a href={profile.github} target="_blank" rel="noopener noreferrer">{profile.githubLabel}</a>
                    </p>
                  </td>
                  <td className="intro-photo">
                    <img alt="Rudramani Singha profile photo" src={profile.photo} className="profile-image" />
                  </td>
                </tr>
              </tbody>
            </table>

            <table className="content-table">
              <tbody>
                <tr>
                  <td className="section-cell">
                    <h2 className="section-title">Selected Works</h2>
                  </td>
                </tr>
              </tbody>
            </table>

            <table className="content-table">
              <tbody>
                <tr className="work-row">
                  <td className="work-content-cell">
                    <h3 className="paper-title">{paper.title}</h3>
                    <PaperVideo />
                    <p className="paper-links">
                      <span>links:</span>
                      <a href={paper.url} target="_blank" rel="noopener noreferrer">
                        <img className="nature-logo" src={paper.natureLogo} alt="Nature" /> <span>{paper.year}</span>
                      </a>
                      <a href={paper.code} target="_blank" rel="noopener noreferrer">
                        <img className="github-logo" src={paper.githubLogo} alt="GitHub" /> <span>code</span>
                      </a>
                    </p>
                    <p className="paper-authors">
                      {paper.authors.map((author, index) => <span key={author}>
                        {index > 0 ? ', ' : ''}<span className={author === profile.name ? 'author-self' : undefined}>{author}</span>{paper.equalContribution?.includes(author) && <sup>*</sup>}
                      </span>)}
                    </p>
                    {paper.equalContribution && <p><small>* Equal contribution</small></p>}
                    <p>
                      {paper.blurb}
                    </p>
                    {paper.abstractLicense && <p><small><a href={paper.abstractLicense.url} rel="license">{paper.abstractLicense.label}</a></small></p>}
                  </td>
                </tr>

              </tbody>
            </table>

            <table className="content-table">
              <tbody>
                <tr>
                  <td className="footer-cell">
                    <br />
                    <p className="footer-note">
                      &copy; 2026 {profile.name}
                    </p>
                  </td>
                </tr>
              </tbody>
            </table>
          </td>
        </tr>
      </tbody>
    </table>
  );
}

const root = createRoot(document.getElementById('root'));
if (import.meta.env.DEV && /^\/prototype\/e\/?$/.test(location.pathname)) {
  import('./prototypes/e/PrototypeE').then(({ default: PrototypeE }) => {
    root.render(<StrictMode><PrototypeE /></StrictMode>);
  });
} else if (import.meta.env.DEV && /^\/prototype\/d\/?$/.test(location.pathname)) {
  import('./prototypes/d/PrototypeD').then(({ default: PrototypeD }) => {
    root.render(<StrictMode><PrototypeD /></StrictMode>);
  });
} else if (import.meta.env.DEV && /^\/prototype\/c\/?$/.test(location.pathname)) {
  import('./prototypes/c/PrototypeC').then(({ default: PrototypeC }) => {
    root.render(<StrictMode><PrototypeC /></StrictMode>);
  });
} else if (import.meta.env.DEV && /^\/prototype\/b\/?$/.test(location.pathname)) {
  import('./prototypes/b/PrototypeB').then(({ default: PrototypeB }) => {
    root.render(<StrictMode><PrototypeB /></StrictMode>);
  });
} else if (import.meta.env.DEV && /^\/prototype\/a\/?$/.test(location.pathname)) {
  import('./prototypes/a/PrototypeA').then(({ default: PrototypeA }) => {
    root.render(<StrictMode><PrototypeA /></StrictMode>);
  });
} else {
  root.render(<StrictMode><LandingPage /></StrictMode>);
}
