# Draw

Deployment wrapper for ExcaliDash at `draw.singha.io`.

Upstream:

- Repository: `https://github.com/ZimengXiong/ExcaliDash`
- Images:
  - `zimengxiong/excalidash-frontend`
  - `zimengxiong/excalidash-backend`

The ExcaliDash source is not vendored here. This folder tracks our deployment choice: image tags, environment variables, routing labels, state path, and backup notes.

## Update

1. Read the upstream release notes.
2. Change `EXCALIDASH_VERSION` in `infra/.env`.
3. Deploy:

```bash
ops/deploy.sh
```

Do not use `latest` for production unless explicitly testing.
