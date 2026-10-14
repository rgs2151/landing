import { neuraljaxBrainAnimations } from '../../shared/neuraljaxlib/animations';
export const { cnn, rnn, autoencoder, gan, transformer } = Object.fromEntries(neuraljaxBrainAnimations.map(asset => [asset.key, asset.Animation]));
