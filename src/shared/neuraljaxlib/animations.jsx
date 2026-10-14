import SVGAnimation from '../SVGAnimation';

const definitions = [
  { id: 'cnn', label: 'CNN', fps: 60, load: () => Promise.all([import('./vendor/cnn/animation'), import('./vendor/cnn/samples')]) },
  { id: 'rnn', label: 'RNN', fps: 60, load: () => Promise.all([import('./vendor/rnn/animation'), import('./vendor/rnn/samples')]) },
  { id: 'autoencoder', label: 'Autoencoder', fps: 60, load: () => Promise.all([import('./vendor/autoencoder/animation'), import('./vendor/autoencoder/samples')]) },
  { id: 'gan', label: 'GAN', fps: 30, load: () => Promise.all([import('./vendor/gan/animation'), import('./vendor/gan/samples')]) },
  { id: 'transformer', label: 'Transformer', fps: 60, load: () => Promise.all([import('./vendor/transformer/animation'), import('./vendor/transformer/samples')]) },
];

export const neuraljaxAnimations = definitions.map(definition => {
  let loaded;
  const asset = { ...definition, load: () => loaded ??= definition.load().then(([render, data]) => ({ render: render.default, data: data.default })) };
  const Animation = props => <SVGAnimation {...props} asset={asset} />;
  return { id: `neuraljax-${asset.id}`, key: asset.id, Animation };
});
