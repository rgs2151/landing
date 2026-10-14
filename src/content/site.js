import neuralFieldManifoldBlurb from './neural-field-manifold.md?raw';
import robustSteeringBlurb from './robust-steering.md?raw';
import natureAbstract from './nature-abstract.md?raw';

export const profile = {
  name: 'Rudramani Singha',
  photo: '/images/hero.jpg',
  email: 'rgs2151@columbia.edu',
  github: 'https://github.com/rgs2151',
  githubLabel: 'github.com/rgs2151',
  statement: 'I build models to understand the brain.',
  role: 'I am a Data Scientist at the',
  institution: 'Program in Memory Longevity',
  institutionUrl: 'https://memorylongevity.org/',
  affiliation: ', UTSW.',
};

export const biography = {
  ...profile,
  role: 'I am a data scientist in the',
  affiliation: ' at UT Southwestern Medical Center.',
  statement: 'I received my M.S. in Biomedical Engineering from Columbia University. My research combines computational neuroscience and machine learning to study neural dynamics and information processing in biological and artificial neural networks.',
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
  equalContribution: ['Yuan Cheng', 'Yusi Chen'],
  blurb: natureAbstract.trim(),
  abstractLicense: { label: 'CC BY-NC-ND 4.0', url: 'https://creativecommons.org/licenses/by-nc-nd/4.0/' },
  clips: [6, 7, 8, 9].map(number => ({
    src: `/assets/videos/41586_2026_10900_MOESM${number}_ESM.mp4`,
    poster: `/assets/images/paper-${number}.webp`,
  })),
  strip: '/assets/videos/paper-wide.mp4',
  stripPoster: '/assets/videos/paper-wide.jpg',
  natureLogo: '/assets/logos/nature.svg',
  githubLogo: '/assets/logos/github.svg',
};

export const robustSteering = {
  id: 'robust-steering',
  shortTitle: 'Robust steering',
  animation: {
    type: 'javascript',
    width: 776,
    height: 550,
    alt: 'Illustrative robust steering animation: a controller learned from one activation landscape is applied after a distribution shift to steer a response into Spanish.',
  },
  title: 'Mind Control of LLMs via Closed-Loop Stimulation: Out-of-Distribution Robustness',
  url: 'https://rgs2151.github.io/controller/',
  code: 'https://github.com/rgs2151/controller',
  year: '2026',
  status: 'Under review',
  authors: ['Kasra Fallah', 'Rudramani Singha', 'Hannah Ghanei', 'Paul Sajda', 'James Anderson', 'Erfan Zabeh'],
  equalContribution: ['Kasra Fallah', 'Rudramani Singha'],
  reference: 'design/references/papers/llm-robust-stimulation.pdf',
  blurb: robustSteeringBlurb.trim(),
};

export const neuralFieldManifold = {
  id: 'neural-field-manifold',
  title: 'NeuralFieldManifold: Reconstruction of LFP manifold with Lag Embedding',
  url: 'https://erfanzabeh.github.io/NeuralFieldManifold/',
  code: 'https://github.com/erfanzabeh/NeuralFieldManifold',
  year: '2026',
  status: 'Accepted',
  venue: 'NeurIPS',
  authors: ['Kasra Fallah', 'Haoyu Novak Chen', 'Rudramani Singha', 'Eunji Kong', 'Gergely Turi', 'Attila Losonczy', 'Erfan Zabeh'],
  reference: 'design/references/papers/neural-field-manifold.pdf',
  blurb: neuralFieldManifoldBlurb.trim(),
};

export const projects = [
  { id: 'cooperation', ...paper },
  robustSteering,
  neuralFieldManifold,
];
