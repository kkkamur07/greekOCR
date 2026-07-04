# TODO

- Think through the `ml/` directory structure before it grows further. In particular, clarify the boundary between public API/job contracts in `ml/contracts/` and internal model registry schema/loaders in `ml/registry/`; registry-specific types such as architecture/device probably should not live in a generic contracts schema.
- Design the future remote weight source flow separately from local `file://` registry resolution. Local paths should stay constrained under `ML_ROOT`; server-backed weights should go through an explicit downloader/cache layer with allowed schemes, integrity checks, and clear cache layout.
- Use celery and `BackgroundTasks` to handle the background tasks later on. 
