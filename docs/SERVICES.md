# Services

| Domain | Folder | Type | Runtime | State |
| --- | --- | --- | --- | --- |
| `singha.io` | `apps/landing/` | owned app | React/Vite built into nginx image | none |
| `draw.singha.io` | `services/draw/` | third-party service | ExcaliDash frontend/backend images | `/srv/singha/state/draw/backend` |

## Add A Service

1. Create `services/<name>/`.
2. Add `compose.yaml` with pinned images and Traefik labels.
3. Add `.env.example`.
4. Add `backup.md` with exact state paths and restore procedure.
5. Add it to `infra/compose.yaml` under `include`.
6. Add the domain to this table.
