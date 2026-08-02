# Restore

The repository restores code and deployment configuration. Runtime data restores from restic.

## Code

```bash
git clone https://github.com/rgs2151/landing.git /srv/singha/repo
cd /srv/singha/repo
git switch expansion
```

## State

Restore service data into:

```text
/srv/singha/state
```

Each service folder owns its restore notes. Start with:

- `services/draw/backup.md`

## Bring Services Back

```bash
ops/deploy.sh
```
