# Draw Backup

ExcaliDash currently stores SQLite state under:

```text
/srv/singha/state/draw/backend
```

This path must be included in restic backups.

For a conservative restore test:

1. Stop the draw services.
2. Restore `/srv/singha/state/draw/backend`.
3. Start the draw services.
4. Confirm login and drawing list load at `draw.singha.io`.
