# dices-ai

FastAPI service that detects dice in uploaded images using a YOLO model and renders the results (annotated images, counts, and a per-face distribution) as HTML.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management
- A trained YOLO weights file at `./model/best.pt` (override with `MODEL_PATH`)

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn src.main:app --reload
```

The app is then available at <http://127.0.0.1:8000>. Open `/` for the upload UI.

## Configuration

Settings are loaded via `pydantic-settings` from environment variables (see [src/config.py](src/config.py)).

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `./model/best.pt` | Path to the YOLO weights file |
| `MAX_FILES_COUNT` | `100` | Upper bound on uploads per request |

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Upload form |
| POST | `/dices/check` | Single image → annotated image with bounding boxes |
| POST | `/dices/stats` | Multiple images → total image and dice counts |
| POST | `/dices/distribution` | Multiple images → per-face count and standard deviation |

Allowed image extensions: `.jpg`, `.jpeg`, `.png` (max 10 MB each — see [src/validator.py](src/validator.py)).

## Development

```bash
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run ty check            # type check
uv run pre-commit install  # enable git hooks
```

CI runs lint, type-check, and tests on every push — see [.github/workflows/ci.yaml](.github/workflows/ci.yaml).

## Project layout

```
src/
  main.py        # FastAPI app
  routes.py      # HTTP endpoints
  service.py     # YOLO inference, drawing, distribution math
  validator.py   # Upload validation
  schemas.py     # Pydantic models
  config.py      # Settings
  templates/     # Jinja2 templates
tests/           # Pytest suite
model/best.pt    # YOLO weights (not committed)
```
