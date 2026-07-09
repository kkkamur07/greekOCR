# TODO

- Move logging to [loguru](https://github.com/Delgan/loguru) across the Python services (`nomicous/backend`, `inference`, `ml/`, scripts). Replace stdlib `logging` and SQLAlchemy `echo=True` dev noise with structured, level-controlled loguru output.
- Hugging face integration as a model repository so that It works from everywhere so that It functions as a repository from where we can just pull the runtime models like kraken and calamari. ( in process )
- Need UI and UX improvements like not intuitive to add the points for the polygons and I think we are being too strict on polygon approximation. 
- Take a look at the notification system please, because it something really important -> too many API calls there. 
- **Targeted API mutations (issue 037):** Page editor was using `PUT /lines` (full replace) for single-segment delete/draw/resize. Partial fix in branch — see `issues/037-targeted-api-mutations-audit.md` for audit, remaining tests, and other full-reload hotspots.
- Need to do docker build optimization, the apis are consuming a lot of space most likely because of tensorflow. 

- To be honest looks really good and performs really well just some minor fixes and we will be good to go. but to understand what is meant with good code we need to first build our reference base by reading good code realted to this, don't really care about the frontend at the moment but the backend and infrasturcture need to learn the best practices first principles wise. 
termi
- Supabase migration for quick current deployment, rest we can quickly deploy the services on vercel. 