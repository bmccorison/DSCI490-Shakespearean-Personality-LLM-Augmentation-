# Shakespearean Interface

Simple React + Tailwind frontend for the FastAPI endpoints in `app.py`.

## Run

```bash
cd interface
npm install
npm run dev
```

Backend expected at `http://127.0.0.1:8000` and served under the `/api` namespace.

## Endpoint Usage

- `GET /api/generate_response` on message send
- `GET /api/refresh_chat` on chat reset and startup
- `GET /api/select_character` on character setup
- `GET /api/select_model` on model setup
- `GET /api/get_models` on startup and model reload
- `GET /api/multimodel/config` on startup for conversation limits
- `POST /api/multimodel/config` when saving the model-conversation turn limit
- `POST /api/multimodel/start` when starting a model-to-model session
- `POST /api/multimodel/next` for each generated model-to-model turn
- `POST /api/multimodel/stop` when stopping a running model conversation
- `POST /api/tts` on assistant voice playback
