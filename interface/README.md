# Shakespearean Interface

Simple React + Tailwind frontend for the FastAPI endpoints in `app.py`.

## Run

```bash
cd interface
npm install
npm run dev
```

Backend expected at `http://127.0.0.1:8000` (proxied as `/api` in development).

## Endpoint Usage

- `GET /generate_response` on message send
- `GET /refresh_chat` on chat reset and startup
- `GET /select_character` on character setup
- `GET /select_model` on model setup
- `GET /get_models` on startup and model reload
- `POST /tts` on assistant voice playback
