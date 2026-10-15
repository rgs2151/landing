import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { dirname } from 'node:path';

const projects = ['neuraljaxlib', 'neural-field-manifold', 'robust-steering', 'pylba'];
const hash = file => createHash('sha256').update(readFileSync(file)).digest('hex');
const selected = process.env.ASSETS?.split(',');
for (const project of projects) {
  const folder = `public/assets/projects/${project}`;
  const previews = JSON.parse(readFileSync(`${folder}/previews.json`));
  let mobile = {};
  try { mobile = JSON.parse(readFileSync(`${folder}/mobile-previews.json`)); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  for (const clip of Object.values(previews)) {
    if (selected && !selected.includes(clip.id)) continue;
    const width = Math.min(clip.width, 480);
    const height = Math.round(clip.height * width / clip.width / 2) * 2;
    const src = `/assets/projects/${project}/videos/mobile/${clip.name}.mp4`;
    mkdirSync(dirname(`public${src}`), { recursive: true });
    // H.264 has no alpha: pack RGB on the left and its grayscale alpha on the
    // right. The phone player restores transparency without rerunning simulations.
    execFileSync('ffmpeg', ['-y', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', `public${clip.src}`,
      '-filter_complex', `[0:v]scale=${width}:${height}:flags=lanczos,format=yuva420p,split[color][alpha];[color]format=yuv420p[rgb];[alpha]alphaextract,format=yuv420p[mask];[rgb][mask]hstack`,
      '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '16', '-pix_fmt', 'yuv420p',
      '-movflags', '+faststart', `public${src}`], { stdio: 'inherit' });
    mobile[clip.id] = { src, width, height, encoding: 'rgb-alpha-side-by-side',
      revision: hash(`public${src}`).slice(0, 12), source: clip.src, sourceSha256: hash(`public${clip.src}`),
      bytes: statSync(`public${src}`).size };
    console.log(`${clip.id}: ${width}x${height}, ${mobile[clip.id].bytes} bytes`);
  }
  writeFileSync(`${folder}/mobile-previews.json`, `${JSON.stringify(mobile, null, 2)}\n`);
}
