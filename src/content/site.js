export const profile = {
  name: 'Rudramani Singha',
  photo: '/images/hero.jpg',
  email: 'rgs2151@columbia.eu',
  github: 'https://github.com/rgs2151',
  githubLabel: 'github.com/rgs2151',
  statement: 'I build models to understand the brain.',
  role: 'I am a Data Scientist at the',
  institution: 'Program in Memory Longevity',
  institutionUrl: 'https://memorylongevity.org/',
  affiliation: ', UTSW.',
};

export const paper = {
  title: 'Asymmetric prefrontal representations for leader\u2013follower dynamics',
  url: 'https://www.nature.com/articles/s41586-026-10900-1',
  code: 'https://github.com/NuttidaLab/MARLAX',
  year: '2026',
  authors: [
    'Yuan Cheng', 'Yusi Chen', 'Myungji Kwak', 'Ross P. Kempner',
    'Rudramani Singha', 'Jared Winslow', 'Runqi Liu', 'Umais Khan',
    'Tessa Spangler', 'Alvi Khan', 'Talmo Pereira', 'Matthew Whiteway',
    'Evan S. Schaffer', 'Nuttida Rungratsameetaweemana', 'Nan Yang',
    'Herbert Zheng Wu',
  ],
  blurb: 'We introduce a mouse paradigm to study cooperative behavior where stable leader-follower roles emerge during joint foraging. Using calcium imaging and optogenetic disruption, the study shows medial prefrontal cortex representations are role-specific and critical for cooperation. I developed the forward-modeling framework paired with multi-agent inverse reinforcement learning to decode latent value functions driving cooperative decisions.',
  clips: [6, 7, 8, 9].map(number => ({
    src: `/assets/videos/41586_2026_10900_MOESM${number}_ESM.mp4`,
    poster: `/assets/images/paper-${number}.webp`,
  })),
  strip: '/assets/videos/paper-wide.mp4',
  stripPoster: '/assets/videos/paper-wide.jpg',
  natureLogo: '/assets/logos/nature.svg',
  githubLogo: '/assets/logos/github.svg',
};

export const llmSteering = {
  id: 'llm-steering',
  title: 'Mind Control of LLMs via Closed-Loop Stimulation: Out-of-Distribution Robustness',
  url: 'https://rgs2151.github.io/controller/',
  code: 'https://github.com/rgs2151/controller',
  year: '2026',
  status: 'Manuscript',
  authors: ['Kasra Fallah', 'Rudramani Singha', 'Hannah Ghanei', 'Paul Sajda', 'James Anderson', 'Erfan Zabeh'],
  equalContribution: ['Kasra Fallah', 'Rudramani Singha'],
  reference: 'design/references/papers/llm-robust-stimulation.pdf',
  blurb: 'We formulate LLM steering as a feedback-control problem inspired by closed-loop neurostimulation. Our robust controller adapts interventions to the model\'s evolving internal state while accounting for uncertainty in its dynamics, improving steering under shifts in language, adversarial prompts, and context length without changing model weights. We also introduce robust steerability, a pre-deployment measure of how much dynamics mismatch an intervention can tolerate, connecting reliable control in artificial and biological neural systems.',
};

export const neuralFieldManifold = {
  id: 'neural-field-manifold',
  title: 'NeuralFieldManifold: Reconstruction of LFP manifold with Lag Embedding',
  url: 'https://erfanzabeh.github.io/NeuralFieldManifold/',
  code: 'https://github.com/erfanzabeh/NeuralFieldManifold',
  year: '2026',
  status: 'Preprint',
  authors: ['Kasra Fallah', 'Haoyu Novak Chen', 'Rudramani Singha', 'Eunji Kong', 'Gergely Turi', 'Attila Losonczy', 'Erfan Zabeh'],
  reference: 'design/references/papers/neural-field-manifold.pdf',
  blurb: 'We develop a framework for recovering the geometry of local field potentials, revealing the low-dimensional structure behind population-level brain signals. Under an autoregressive model, we show how sustained oscillatory modes give rise to toroidal manifolds through lag embedding, while noise and aperiodic activity contribute geometric thickness. We further propose a geometry-aware learning network for time-varying dynamics and validate our framework on synthetic systems and neural recordings across species.',
};

export const projects = [
  { id: 'cooperation', ...paper },
  llmSteering,
  neuralFieldManifold,
];
