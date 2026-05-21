### Greek OCR

Main goal of the project is to **Make it easier for greek byzantine researchers to transcribe the greek manuscripts** 

This is going to be a difficult task probably going to take months but the following outcomes can be expected : 

1. Novel Legal Manuscripts data curation $\to$ which can be made publically available
2. VLM Models & OCR models specifically for *greek* which can be used for various purposes
3. Platform for data annotation with expert in the loop to get accurate labels. 

This is a very novel work and can enhance the quality of work everywhere in the world. 

## Platform development (Postgres + DDD backend)

The new platform API lives under `backend/core/` with bounded contexts (`users`, `project`, `document`, `inference`). Postgres and Alembic are in repo-root `infrastructure/`.

### Prerequisites

- Python 3.11+
- Docker (for Postgres and optional full stack)

### Quick start (Docker Compose)

```bash
cp backend/core/.env.example backend/core/.env
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Health | http://localhost:8000/health |
| OpenAPI | http://localhost:8000/docs |
| Postgres | `localhost:5433` (user `postgres`, password `dev`, db **`kalamos`**) |

See [infrastructure/README.md](infrastructure/README.md) (DB/migrations) and [backend/core/README.md](backend/core/README.md) (settings, routes, DTOs).

Migrations run automatically on API container start (`alembic upgrade head`).

### Local API without Docker (DB in Docker only)

```bash
docker compose up db -d
uv venv && source .venv/bin/activate
uv pip install -r requirements-platform.txt
export PYTHONPATH=.
cp backend/core/.env.example backend/core/.env
alembic -c infrastructure/alembic.ini upgrade head
uvicorn backend.core.main:app --reload --host 0.0.0.0 --port 8000
pytest
```

**Postgres (local dev):** user `postgres`, password **`dev`**, database **`kalamos`**, port `5433`.

---

## Legacy prototype (Vite + upload API)

Run the **legacy FastAPI backend** and **Vite (React) frontend** in two separate terminals. The UI is configured to call the API at `http://localhost:8000` (see `frontend/src/services/api.ts`). CORS on the API allows `http://localhost:5173` and `http://localhost:3000`.

### Backend (legacy prototype)

```bash
uv sync
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health check: [http://localhost:8000/](http://localhost:8000/)
- Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)



### Frontend (React + Vite)

```bash
cd /path/to/greek-foundation/frontend
npm install
npm run dev
```

Open the URL printed in the terminal (by default [http://localhost:5173/](http://localhost:5173/)).

---

My **current priorites** are : 

1. Figuring out a good base line $\to$ which means experimenting with different providers, open source models to see what is the accuracy we are getting and using that baseline to : 
   1. Improve the upcoming models
   2. Make data annotation and labellling as easy as possible so that we can    
      1. Organize the data 
      2. Create a vast repository of annotated labels

---

*Thoughts* : I have been thinking of experimenting with googleOCR models via their APIs and using VLM open sourced for greek text identification in zero shot and also using openly available greek OCR models to figure out what is the current state. 

> [!IMPORTANT]
> Create a tool for **comparision** and **labelling** essentially data annotations. 

Have realised one thing that nothing is really working but there is till hope with 
1. `kraken` for segmentation and pre processing and using 
2. finetuned version of `TrOCR` for extracting the text from the segmented image. 

So problem of segmentation is not a problem. The pipeline would like to first binarize and then segment and using the segments we have to use the `TrOCR` models, the current segments are really really good. It uses the base model blla.mlmodel 

I think the `TrOCR` model is using the RoBERTa which is an english only architecture, so the current **cer** and **wer** stands at 0.97 and 1.33 which is really bad. Tried to replace the tokenizer with XLM-RoBERTa the **cer** and **wer** are 1.00 and 1.17 respectively. 

I have added the frontend capability to visualize the segments and my current **cer** is hovering around 0.75

### Next priorities are : 
1. Create a good frontend & Make the frontend better.
2. Filter the bounding boxes we have gotten, analyze them and see which of them are fit for transcription
3. Train the model with the bibleData, esteban game me the script and do the hyper parameter optimization - see how esteban is doing it.
4. If possible host it

--- 

### Updated

I have updated the code, now the frontend is based on react. It is a small powerful editor to edit, which is AI assisted ofcourse, I need to make the editor more powerful. The features in the pipeline are to have a special dialog for the selected transcript area, and mostly related to transcript area. 

I also need to train the model, to increase the character accuracy. Also need to clear all the AI Slop out there. 

Damm, this will definetly need fine tuning, relevant resources : 
1. [HuggingFace](https://discuss.huggingface.co/t/fine-tuning-trocr-on-new-language/58234)
2. [Github](https://github.com/huggingface/transformers/issues/19329)
3. [FineTuning](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb)
4. [FineTuningOther](https://github.com/microsoft/unilm/issues/627)


### Progress : 
1. Nothing much from my side, just checking the code and experimenting with different ideas. Nothing tangible from my side 
2. What did I do : 1. Mostly busy with exams 2. Tried out what is working and what is not working 3. Learning react & next js so that frontend can be made much better 4. Got some feedback from researchers to implement what is necessary 5. Probably next month will better progress. 

---

### Run Backend and Frontend

Run both services in separate terminals from the project root.

#### 1) Backend (FastAPI)

```bash
# from project root
python3 -m venv .venv
source .venv/bin/activate
uv sync
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

#### 2) Frontend (React + Vite)

```bash
# in a new terminal
cd frontend
npm install
npm run dev
```

Frontend will be available at:
- App: `http://localhost:5173`

#### 3) Verify full stack

1. Open `http://localhost:5173`
2. Upload an image in the UI
3. Confirm backend requests are successful in browser devtools/network tab
2. What did I do : 1. Mostly busy with exams 2. Tried out what is working and what is not working 3. Learning react & next js so that frontend can be made much better 4. Got some feedback from researchers to implement what is necessary 5. Probably next month will better progress. 


### Building / Extending Escriptorium 

We need to move away from django to fastAPI + pydantic for validation, they have specificed the domain models quite well 


  ┌───────────────────┬────────────────────────────────────────────────────────────┐
  │ Entity            │ Role                                                       │
  ├───────────────────┼────────────────────────────────────────────────────────────┤
  │ Project           │ Workspace; sharing via users/groups                        │
  │ Document          │ A manuscript/book; workflow (draft / published / archived) │
  │ DocumentPart      │ One page/image (ordered)                                   │
  │ Block             │ Region on a page (column, illustration, etc.)              │
  │ Line              │ Text line with geometry (baseline polygon)                 │
  │ Transcription     │ Named transcription layer on a document                    │
  │ LineTranscription │ Text + confidence per line                                 │
  │ OcrModel          │ Trained Kraken recognition/segmentation models             │
  └───────────────────┴────────────────────────────────────────────────────────────┘

  Celery is behind to run the kraken / training import -> kind of important to parallelize the work and have them do it effectively for heavy tasks. 

  FastAPI till sometime can handle celery as well, shouldn't be a problem. 
  The stack for us should be ideally : 
  1. Orchestration : Docker 
  2. Backend : FastAPI + uvicorn, doesn't matter if we use celery but if we see benefits then 100% we will. 
  3. Frontend : NextJS / React (we will need to reuse a lot fo components because it has all the nuances of it. ) -> Reuse the frontend the most and backend we can leave like that. • UI libs: Bootstrap 4, Annotorious, Recogito, Paper.js, axios, reconnecting WebSocket should be migrated to nextJS ( we need to do this component wise because I am no expert in UI )
  4. Postgres is good but we need something to handle the migrations like alembic -> like the docker setup we had. 
  5. Backend should expose the segment, trascibe end points so that we can easily integrate models from hugging face and use our models. 
  6. Authentication : JWT should be good enough for a small scale project but we can scale this up if required. 
  7. Monitoring : Not a priority right now. 

  We are going to do domain driven design : 
  /backend
   /core - FastAPI app entry, router wiring, shared dependencies
   /infrastructure (repo root) - SHARED: Postgres (db.py), config, Alembic, models aggregator for migrations
   /users, /project, /document, /inference — each with:
      /domain
      /application
      /infrastructure  # context ORM/repos only — not the global DB engine
      /api

This will make the project quite maintainable for me and extendable in future. We are going to use Test Driven Development, currently the tests are not a priority but : If we are testing then we are going to test with 1. One failing examples for each case 2. Multiple passing examples. 

Frontend forms should validate user input client-side and surface API `detail` messages from the backend for server-side failures.