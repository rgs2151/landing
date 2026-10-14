import { pylba } from '../content/pylba';
import { neuraljaxlib } from '../content/neuraljaxlib';
import PyLBAAnimation from './pylba/PyLBAAnimation';
import { neuraljaxAnimations } from './neuraljaxlib/animations';

// Shared tool schema: metadata plus reusable figures paired with source content.
export const tools = [
  { ...pylba, figures: [{ id: 'pylba', Animation: PyLBAAnimation, text: pylba.blurb, citation: pylba.citation }] },
  { ...neuraljaxlib, figures: neuraljaxAnimations.map((asset, index) => ({
    ...asset,
    text: [index === 0 ? neuraljaxlib.introduction : '', neuraljaxlib.descriptions[asset.key]].filter(Boolean).join('\n\n'),
  })) },
];
