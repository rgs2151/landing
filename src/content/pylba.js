import blurb from './pylba.md?raw';
import { profile } from './site';

export const pylba = {
  id: 'pylba',
  title: '\u{1f680} PyLBA',
  shortTitle: 'PyLBA',
  authors: [profile.name],
  url: 'https://rgs2151.github.io/pyLBA/',
  code: 'https://github.com/rgs2151/pyLBA',
  blurb: blurb.trim(),
  citation: { text: 'Brown & Heathcote, 2008', url: 'https://doi.org/10.1016/j.cogpsych.2007.12.002' },
};
