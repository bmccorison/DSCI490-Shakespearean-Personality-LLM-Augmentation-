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
- `POST /api/tts` on assistant voice playback
