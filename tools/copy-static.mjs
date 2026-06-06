import { copyFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const distDir = 'dist';
const files = ['404.html', 'CNAME', '.nojekyll', 'stylesheet.css', 'favicon.ico'];

await mkdir(distDir, { recursive: true });

await Promise.all(
  files.map((file) => copyFile(file, join(distDir, file)))
);
