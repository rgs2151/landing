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
