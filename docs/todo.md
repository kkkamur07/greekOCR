# TODO

- Move logging to [loguru](https://github.com/Delgan/loguru) across the Python services (`nomicous/backend`, `inference`, `ml/`, scripts). Replace stdlib `logging` and SQLAlchemy `echo=True` dev noise with structured, level-controlled loguru output.
- Fix noisy Docker health checks: `docker-compose.yml` polls `/health` every 5s from inside each container, and the api health route runs `SELECT 1` (with SQLAlchemy `echo=True` in dev), flooding logs. Consider a lightweight liveness endpoint without DB, turning off SQL echo for health traffic, longer intervals in dev, and/or filtering health-check access logs.
- Think through the `ml/` directory structure before it grows further. In particular, clarify the boundary between public API/job contracts in `ml/contracts/` and internal model registry schema/loaders in `ml/registry/`; registry-specific types such as architecture/device probably should not live in a generic contracts schema.
- Design the future remote weight source flow separately from local `file://` registry resolution. Local paths should stay constrained under `INFERENCE_ROOT`; server-backed weights should go through an explicit downloader/cache layer with allowed schemes, integrity checks, and clear cache layout.
- Longer term plan is to port the Calamari model and forward pass into a first-class `inference/architectures/` implementation (see [`docs/calamari-vendored-architecture.md`](docs/calamari-vendored-architecture.md#future-work)) to reduce dependence on the full TensorFlow stack — especially for faster `linux/arm64` inference images. Until then, vendored source lives in `src/model/calamari` and is copied to `/app/_support_repo/calamari` in the inference Docker image.
- Hugging face integration as a model repository so that It works from everywhere so that It functions as a repository from where we can just pull the runtime models like kraken and calamari. 
- Need UI and UX improvements like not intuitive to add the points for the polygons and I think we are being too strict on polygon approximation. 
- Status draft is not aligned well, need to fix that. 
- Transcription should only be the ground truth there shouldn't like model transcription <id> it is very unintuitive. but again this is good for history to be honest okay the PDF doesn't have support for syriac to be honest. 
- Also in the projects page where it is written jobs/ you should show the history of jobs related to that project and the current jobs that are running or not. 
- In the public view link everything is looking cool just you forgot to essentially scale the segments polygon relatively and also add an option of pdfs ( which are transcribed just like the one we have in there. )
- Take a look at the notification system please, because it something really important -> too many API calls there. 

