import { StrictMode, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { createRoot } from 'react-dom/client';

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
                  <td className="work-image-cell">
                    <img src="images/marlax.gif" alt="Cooperative mouse behavior" className="work-thumb" />
                  </td>
                  <td className="work-content-cell">
                    <p className="paper-venue"><strong><em>Nature</em></strong> <span>2026</span></p>
                    <a href="https://www.nature.com/articles/s41586-026-10900-1" target="_blank" rel="noopener noreferrer">
                      <span className="paper-title">Asymmetric prefrontal representations for leader&ndash;follower dynamics</span>
                    </a>
                    <p className="paper-authors">
                      Yuan Cheng, Yusi Chen, Myungji Kwak, Ross P. Kempner, <strong>Rudramani Singha</strong>, Jared Winslow, Runqi Liu, Umais Khan, Tessa Spangler, Alvi Khan, Talmo Pereira, Matthew Whiteway, Evan S. Schaffer, Nuttida Rungratsameetaweemana, Nan Yang, Herbert Zheng Wu
                    </p>
                    <p className="paper-links">
                      <em>links:</em> [<a href="https://www.nature.com/articles/s41586-026-10900-1" target="_blank" rel="noopener noreferrer">paper</a>] [<a href="https://github.com/NuttidaLab/MARLAX" target="_blank" rel="noopener noreferrer">code</a>]
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
