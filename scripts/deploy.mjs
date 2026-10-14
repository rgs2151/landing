import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const git = (args, cwd = root) => execFileSync('git', args, { cwd, encoding: 'utf8' }).trim();

if (git(['branch', '--show-current']) !== 'expansion') {
  throw new Error('Deploy from the expansion source branch.');
}
if (git(['status', '--porcelain'])) {
  throw new Error('Commit your source changes before deploying.');
}

const source = git(['rev-parse', '--short', 'HEAD']);
const remote = git(['remote', 'get-url', '--push', 'origin']);
const name = git(['var', 'GIT_AUTHOR_IDENT']).match(/^(.*?) <(.*?)>/);
const published = git(['ls-remote', '--heads', remote, 'refs/heads/deploy']);
execFileSync('npm', ['run', 'build'], { cwd: root, stdio: 'inherit' });
const dist = join(root, 'dist');
if (!existsSync(join(dist, 'index.html')) || !existsSync(join(dist, 'CNAME'))) {
  throw new Error('Build must contain index.html and the custom-domain CNAME.');
}

// Publish in an isolated checkout, preserving deployment history and the source tree.
const staging = mkdtempSync(join(tmpdir(), 'landing-deploy-'));
try {
  git(['init', '--initial-branch=deploy', staging]);
  git(['config', 'user.name', name[1]], staging);
  git(['config', 'user.email', name[2]], staging);
  git(['remote', 'add', 'origin', remote], staging);
  if (published) {
    git(['fetch', 'origin', 'deploy'], staging);
    git(['checkout', '-B', 'deploy', 'FETCH_HEAD'], staging);
  }
  for (const entry of readdirSync(staging)) {
    if (entry !== '.git') rmSync(join(staging, entry), { recursive: true, force: true });
  }
  cpSync(dist, staging, { recursive: true });
  writeFileSync(join(staging, '.nojekyll'), '');
  git(['add', '--all'], staging);
  if (git(['status', '--porcelain'], staging)) {
    git(['commit', '-m', `Deploy website from expansion ${source}`], staging);
    execFileSync('git', ['push', 'origin', 'HEAD:refs/heads/deploy'], {
      cwd: staging, stdio: 'inherit',
    });
    console.log('Published to deploy. GitHub Pages will publish this build; domain routing is configured separately.');
  } else {
    console.log('The deploy branch already matches this build.');
  }
} finally {
  rmSync(staging, { recursive: true, force: true });
}
