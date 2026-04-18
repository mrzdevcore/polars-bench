# AI Polars Code Generator

## Overview

A FastAPI backend service that uses the **Qwen2.5-Coder-3B-Instruct** LLM to generate Python Polars code from natural language queries.

## Tech Stack

- **Language:** Python 3.12
- **Web Framework:** FastAPI + Uvicorn
- **AI/ML:** PyTorch (CPU), Hugging Face Transformers, Accelerate
- **Model:** `Qwen/Qwen2.5-Coder-3B-Instruct` (~6GB, cached in `.cache/`)

## Project Layout

```
main.py          - FastAPI app with /health and /chat endpoints
pyproject.toml   - Project metadata and dependencies
requirements.txt - Flat dependency list
```

## API Endpoints

- `GET /` — Health check, returns `{"status": "ok"}`
- `POST /chat` — Accepts `{"message": str, "tables": dict}`, returns `{"response": str}` with generated Polars code

## Running

The app runs via workflow: `uvicorn main:app --host 0.0.0.0 --port 8000`

- First startup downloads the model (~6GB). Subsequent starts load from cache and take ~20 seconds.

## Notes

- Model runs in float16 precision with `device_map="auto"` (CPU in this environment)
- No GPU required but inference will be slow on CPU
