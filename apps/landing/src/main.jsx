import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

function LandingPage() {
  return (
    <table className="page-shell">
      <tbody>
        <tr className="row-reset">
          <td className="cell-reset">
            <table className="content-table">
              <tbody>
                <tr className="row-reset intro-row">
                  <td className="intro-text">
                    <p className="text-center">
                      <span className="site-name">Rudramani Singha</span>
                    </p>
                    <p>
                      I am a Data Scientist at the <a href="https://memorylongevity.org/" target="_blank" rel="noopener noreferrer">Program in Memory Longevity</a>, UTSW.
                      I build probabilistic models to understand the brain.
                    </p>
                    <p className="text-center">
                      <a href="mailto:rgs2151@columbia.eu">rgs2151[at]columbia.eu</a> &nbsp;/&nbsp;
                      <a href="https://github.com/rgs2151" target="_blank" rel="noopener noreferrer">GitHub</a> &nbsp;/&nbsp;
                      <a href="https://scholar.google.com/citations?user=nN4ARxkAAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Google Scholar</a>
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
                    <h2 className="section-title">Selected Projects</h2>
                  </td>
                </tr>
              </tbody>
            </table>

            <table className="content-table">
              <tbody>
                <tr className="work-row">
                  <td className="work-image-cell">
                    <img src="images/marlax.gif" alt="Asymmetric Social Representations in the Prefrontal Cortex for Cooperative Behavior" className="work-thumb" />
                  </td>
                  <td className="work-content-cell">
                    <a href="https://doi.org/10.1101/2025.08.27.672249" target="_blank" rel="noopener noreferrer">
                      <span className="paper-title">Asymmetric Social Representations in the Prefrontal Cortex for Cooperative Behavior</span>
                    </a>
                    <p className="paper-authors">
                      Yuan Cheng, Yusi Chen, Myungji Kwak, Ross P. Kempner, <strong>Rudramani Singha</strong>, Jared Winslow, Runqi Liu, Umais Khan, Tessa Spangler, Alvi Khan, Talmo Pereira, Matthew Whiteway, Evan S. Schaffer, Nuttida Rungratsameetaweemana, Nan Yang, Herbert Zheng Wu
                    </p>
                    <p className="paper-links">
                      <em>links:</em> [<a href="https://doi.org/10.1101/2025.08.27.672249" target="_blank" rel="noopener noreferrer">bioRxiv</a>] [<a href="https://github.com/NuttidaLab/MARLAX" target="_blank" rel="noopener noreferrer">code</a>]
                    </p>
                    <p>
                      We introduce a mouse paradigm to study cooperative behavior where stable leader-follower roles emerge during joint foraging. Using calcium imaging and optogenetic disruption, the study shows medial prefrontal cortex representations are role-specific and critical for cooperation. I developed the forward-modeling framework paired with multi-agent inverse reinforcement learning to decode latent value functions driving cooperative decisions.
                    </p>
                  </td>
                </tr>

                <tr className="work-row">
                  <td className="work-image-cell">
                    <img src="images/hmm.gif" alt="Bayesian Modeling Tutorial" className="work-thumb" />
                  </td>
                  <td className="work-content-cell">
                    <a href="https://colab.research.google.com/github/rgs2151/landing/blob/master/notebooks/hmm.ipynb" target="_blank" rel="noopener noreferrer">
                      <span className="paper-title">Scaling Up Bayesian Models: Regressions, Mixtures, HMMs, and GLM-HMMs</span>
                    </a>
                    <p className="paper-authors">
                      <strong>Rudramani Singha</strong>
                    </p>
                    <p className="paper-links">
                      <em>links:</em> [<a href="https://colab.research.google.com/github/rgs2151/landing/blob/master/notebooks/hmm.ipynb" target="_blank" rel="noopener noreferrer">website</a>] [<a href="https://example.com" target="_blank" rel="noopener noreferrer">code</a>]
                    </p>
                    <p>
                      This tutorial starts with intuitive Bayesian updates and builds to Hidden Markov Models and related latent-variable methods. It includes GLMs, input-driven Gaussian mixture models, and GLM-HMMs with comparisons between MCMC and EM estimation. Implementations cover PyMC, Stan, NumPyro, JAX, and Dynamax.
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
