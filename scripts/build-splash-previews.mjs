import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';

const directory = new URL('../public/assets/images/previews/', import.meta.url);
mkdirSync(directory, { recursive: true });
const clips = [];
for (const number of [6, 7, 8, 9]) {
  const size = { 6: 320, 7: 160, 8: 128, 9: 96 }[number];
  const source = new URL(`../public/assets/videos/41586_2026_10900_MOESM${number}_ESM.mp4`, import.meta.url);
  const target = new URL(`paper-${number}.webp`, directory);
  const probe = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,r_frame_rate', '-of', 'json', source.pathname], { encoding: 'utf8' });
  if (probe.status !== 0) throw new Error(`Cannot inspect clip ${number}`);
  const stream = JSON.parse(probe.stdout).streams[0];
  const [numerator, denominator] = stream.r_frame_rate.split('/').map(Number);
  const frames = Number(stream.nb_frames), fps = numerator / denominator;
  if (!(frames > 0 && fps > 0)) throw new Error(`Invalid frame metadata: ${number}`);
  const result = spawnSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-i', source.pathname,
    '-vf', `scale=${size}:${size}:flags=lanczos,tile=6x${Math.ceil(frames / 6)}`, '-frames:v', '1', '-c:v', 'libwebp', '-quality', '80', target.pathname], { stdio: 'inherit' });
  if (result.status !== 0) throw new Error(`Preview encoding failed: ${number}`);
  const revision = createHash('sha256').update(readFileSync(target)).digest('hex').slice(0, 12);
  clips.push({ src: `/assets/images/previews/paper-${number}.webp`, frames, fps, size, columns: 6, revision });
}
writeFileSync(new URL('clips.json', directory), JSON.stringify(clips, null, 2) + '\n');
