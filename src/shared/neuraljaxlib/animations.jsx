import SVGAnimation from '../SVGAnimation';

const definitions = [
  { id: 'cnn', label: 'CNN', fps: 60, load: () => Promise.all([import('./vendor/cnn/animation'), import('./vendor/cnn/samples')]) },
  { id: 'rnn', label: 'RNN', fps: 60, load: () => Promise.all([import('./vendor/rnn/animation'), import('./vendor/rnn/samples')]) },
  { id: 'autoencoder', label: 'Autoencoder', fps: 60, load: () => Promise.all([import('./vendor/autoencoder/animation'), import('./vendor/autoencoder/samples')]) },
  { id: 'gan', label: 'GAN', fps: 30, load: () => Promise.all([import('./vendor/gan/animation'), import('./vendor/gan/samples')]) },
  { id: 'transformer', label: 'Transformer', fps: 60, load: () => Promise.all([import('./vendor/transformer/animation'), import('./vendor/transformer/samples')]) },
];

function createAnimation(definition) {
  let loaded;
  const asset = { ...definition, load: () => loaded ??= definition.load().then(([render, data]) => ({ render: render.default, data: data.default })) };
  const Animation = props => <SVGAnimation {...props} asset={asset} />;
  const key = asset.key || asset.id;
  return { id: `neuraljax-${key}`, key, Animation };
}

export const neuraljaxAnimations = definitions.map(createAnimation);

const brainReplacements = Object.fromEntries([
  { key: 'cnn', id: 'cnn-brain-mri', label: 'CNN', fps: 60, load: () => Promise.all([import('./vendor/cnn_brain_mri/animation'), import('./vendor/cnn_brain_mri/samples')]) },
  { key: 'gan', id: 'gan-brain-mri', label: 'GAN', fps: 30, load: () => Promise.all([import('./vendor/gan_brain_mri/animation'), import('./vendor/gan_brain_mri/samples')]) },
].map(definition => [definition.key, createAnimation(definition)]));
export const neuraljaxBrainAnimations = neuraljaxAnimations.map(asset => brainReplacements[asset.key] || asset);

let galleryModule;
const galleryAsset = {
  id: 'gallery', label: 'NeuralJAXlib', fps: 60, designWidth: 1248, controlsInset: true,
  load: () => galleryModule ??= import('./vendor/composite/animation').then(module => ({ render: module.default })),
};
export function NeuralJAXGallery(props) {
  return <SVGAnimation {...props} asset={galleryAsset} />;
}

let brainGalleryModule;
const brainGalleryAsset = {
  ...galleryAsset, id: 'brain-gallery',
  load: () => brainGalleryModule ??= Promise.all([
    import('./vendor/composite/animation'),
    import('./vendor/cnn_brain_mri/animation'), import('./vendor/cnn_brain_mri/samples'),
    import('./vendor/gan_brain_mri/animation'), import('./vendor/gan_brain_mri/samples'),
  ]).then(([gallery, cnn, cnnData, gan, ganData]) => ({
    render: root => gallery.default(root, { cnn: [cnn.default, cnnData.default], gan: [gan.default, ganData.default] }),
  })),
};
export function NeuralJAXBrainGallery(props) {
  return <SVGAnimation {...props} asset={brainGalleryAsset} />;
}
