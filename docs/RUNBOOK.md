# Reproducibility Runbook

Run these commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Generate data:

```bash
python src/generate_data.py
```

Train:

```bash
python src/train.py
```

Test:

```bash
pytest -q
```

Run API:

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The generated artifacts are intentionally separated into `data/`, `models/`, and `results/` so that the experiment can be inspected and reproduced.
