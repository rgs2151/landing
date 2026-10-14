import introduction from './neuraljaxlib.md?raw';
import abstract from './neuraljaxlib-abstract.md?raw';
import logo from '../shared/neuraljaxlib/logo.svg';
import { profile } from './site';

// Source README/gallery descriptions; the accepted abstract is exported below.
export const neuraljaxlib = {
  id: 'neuraljaxlib',
  title: 'NeuralJAXlib',
  shortTitle: 'NeuralJAXlib',
  logo,
  year: 2024,
  authors: [profile.name],
  code: 'https://github.com/rgs2151/NeuralJAXlib',
  url: 'https://rgs2151.github.io/NeuralJAXlib/',
  introduction: introduction.trim(),
  descriptions: {
    cnn: 'A bold red window scans the handwritten digit continuously, alternating direction between rows. Each completed 2\u00d72 patch is replaced in the same place by one larger pixel containing its maximum value. Four passes reduce the padded image from 32\u00d732 to 16\u00d716 to 8\u00d78 to 4\u00d74 to 2\u00d72 while maintaining the square\u2019s physical size.',
    rnn: 'The legend reads Forecast and Real: dark red denotes forecasts, black denotes observations. The data are a fixed, chronological excerpt from January 1\u2013February 18, 2024 (1,176 consecutive hourly samples), exported from the cached NYC weather test predictions. These are archived daily 24-hour forecasts, compared retrospectively with realized temperatures at each target hour\u2014not new one-hour forecasts issued at every animation step. Smooth interpolation between hourly observations is for playback only.',
    autoencoder: 'A compact, approximately 240\u00d7237 composition including its title shows 500 colored test-sample dots, with five representative images at class means, moving between two-dimensional input PCA and bottleneck PCA. These are unsupervised reconstruction models, not five-way classifiers. Labels are used only to select the displayed classes and compute class means.',
    gan: 'Twelve fixed latent inputs cycle in three 2\u00d72 batches through the generator\u2019s training progression. Four samples develop together in each batch. Playback starts at epoch 3, skipping the initial gray frames, and runs at twice the original speed. This shows training progress, not iterative denoising during inference. A trained GAN generates each output in one forward pass.',
    transformer: 'Each new word enters from the right as previous words slide left. The current word is bold and dark red, with the text line close beneath the title. One set of black semicircles below the text shows the mean attention of both heads in transformer layer 3; stronger weights produce thicker arcs. This is word-grouped character attention, not a word-tokenized model.',
  },
};

export const neuraljaxlibRevision = {
  abstract: abstract.trim(),
  resourceLinks: [
    { label: 'Architecture tutorials', url: 'https://rgs2151.github.io/NeuralJAXlib/docs/learning.html' },
    { label: 'API tutorials', url: 'https://rgs2151.github.io/NeuralJAXlib/docs/api_tutorials.html' },
  ],
};
