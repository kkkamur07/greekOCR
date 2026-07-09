# TODO

- Move logging to [loguru](https://github.com/Delgan/loguru) across the Python services (`nomicous/backend`, `inference`, scripts). Replace stdlib `logging` and SQLAlchemy `echo=True` dev noise with structured, level-controlled loguru output.
- **Local inference helper** (issues 038–039, ADR 002): ship slim sidecar + browser-orchestrated `/run` for segment and transcribe; cloud fallback toggle.
- Need UI and UX improvements like not intuitive to add the points for the polygons and I think we are being too strict on polygon approximation.
- To understand what "good code" means for this project, build a reference base by reading strong backend/infrastructure examples (frontend deferred for now).
- We need to add authentication to the sidecars requests in the main production. 
- Need to implement caching and webp only because that is optimized for image rending and these images could be like 20 MB as times so better to store them as webp on supabase. 
- Optimize the side car for macos and windows build as well. 