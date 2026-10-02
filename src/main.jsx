import { StrictMode, useEffect, useRef, useState } from 'react';
import { Moon, Sun, Pause, Play } from 'lucide-react';
import { createRoot } from 'react-dom/client';

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
                      <span className="site-name">Rudramani Singha</span>
                    </p>
                    <p>
                      I am a Data Scientist at the <a href="https://memorylongevity.org/" target="_blank" rel="noopener noreferrer">Program in Memory Longevity</a>, UTSW.
                      I build models to understand the brain.
                    </p>
                    <p className="contact-block text-center">
                      <span>rgs2151@columbia.eu</span>
                      <a href="https://github.com/rgs2151" target="_blank" rel="noopener noreferrer">github.com/rgs2151</a>
                    </p>
                  </td>
                  <td className="intro-photo">
                    <img alt="Rudramani Singha profile photo" src="images/hero.jpg" className="profile-image" />
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
                    <h3 className="paper-title">Asymmetric prefrontal representations for leader&ndash;follower dynamics</h3>
                    <PaperVideo />
                    <p className="paper-links">
                      <span>links:</span>
                      <a href="https://www.nature.com/articles/s41586-026-10900-1" target="_blank" rel="noopener noreferrer">
                        <img className="nature-logo" src="/assets/logos/nature.svg" alt="Nature" /> <span>2026</span>
                      </a>
                      <a href="https://github.com/NuttidaLab/MARLAX" target="_blank" rel="noopener noreferrer">
                        <img className="github-logo" src="/assets/logos/github.svg" alt="GitHub" /> <span>code</span>
                      </a>
                    </p>
                    <p className="paper-authors">
                      Yuan Cheng, Yusi Chen, Myungji Kwak, Ross P. Kempner, <span className="author-self">Rudramani Singha</span>, Jared Winslow, Runqi Liu, Umais Khan, Tessa Spangler, Alvi Khan, Talmo Pereira, Matthew Whiteway, Evan S. Schaffer, Nuttida Rungratsameetaweemana, Nan Yang, Herbert Zheng Wu
                    </p>
                    <p>
                      We introduce a mouse paradigm to study cooperative behavior where stable leader-follower roles emerge during joint foraging. Using calcium imaging and optogenetic disruption, the study shows medial prefrontal cortex representations are role-specific and critical for cooperation. I developed the forward-modeling framework paired with multi-agent inverse reinforcement learning to decode latent value functions driving cooperative decisions.
                    </p>
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
                      &copy; 2026 Rudramani Singha
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

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <LandingPage />
  </StrictMode>
);
