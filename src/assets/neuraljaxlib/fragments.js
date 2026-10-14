import { neuraljaxAnimations } from '../../shared/neuraljaxlib/animations';
export const { cnn, rnn, autoencoder, gan, transformer } = Object.fromEntries(neuraljaxAnimations.map(asset => [asset.key, asset.Animation]));
