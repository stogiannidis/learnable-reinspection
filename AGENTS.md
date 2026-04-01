# Repository Guidelines

## Project Structure & Module Organization
`poc/` contains the synthetic grid proof-of-concept: data generation, baseline vs. re-inspection models, training, evaluation, and figure code. **`reinspection_vlm/`** is the unified Qwen3-VL + InternVL3 training stack (shared module, training, eval, and `backends/` wrappers). Legacy **`reinspection_qwen3vl/`** and **`reinspection_internvl3/`** are thin compatibility shims. Configs live in `reinspection_vlm/configs/{qwen3vl,internvl3}/`; launch scripts in `reinspection_*/scripts/`. `k8s/` holds Kubernetes job manifests, `logs/` stores run logs, and `doc.md` / `simpledoc.md` explain the method and expected outputs.

## Build, Test, and Development Commands
Install only the dependencies for the path you are touching:

- `pip install -r poc/requirements.txt`: install the lightweight PoC stack.
- `python -m poc.run --model both --epochs 30 --seeds 42 123 456`: run the full PoC from Python.
- `bash poc/scripts/run_poc.sh`: run the default PoC experiment.
- `pip install -r reinspection_vlm/requirements.txt`: install the unified VLM training stack (Qwen + InternVL).
- `bash reinspection_qwen3vl/scripts/run_stage1.sh` / `run_stage2.sh` / `run_eval.sh`: Qwen pipelines (call `reinspection_vlm` internally).
- `bash reinspection_internvl3/scripts/run_stage1.sh` / `run_stage2.sh` / `run_eval.sh`: InternVL pipelines.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, `snake_case` for modules/functions, and `PascalCase` for classes such as `ReInspectionConfig`. Follow the existing pattern of short module docstrings, typed helper functions where practical, and runnable CLI entry points via `python -m ...`. Keep imports grouped as standard library, third-party, then local modules. Shell scripts should stay POSIX-friendly and use `set -euo pipefail`.

## Testing Guidelines
No dedicated `tests/` suite is checked in yet. Treat the runnable scripts as smoke tests: rerun the relevant `poc` or `reinspection_qwen3vl` command after changes, and note the dataset root, checkpoint, and GPU count used. If you add tests, place them under a new `tests/` package and name files `test_<area>.py`.

## Commit & Pull Request Guidelines
Local `.git` history is not available in this workspace, so do not infer a hidden convention. Use short, imperative, scoped commit messages such as `poc: tighten attention metric logging` or `qwen3vl: fix stage2 checkpoint loading`. PRs should state the affected path, commands run, required environment variables, and any metric changes or generated figures.

## Security & Configuration Tips
Keep secrets and cluster-specific values out of source-controlled YAML. Pass `DATA_ROOT`, `OUTPUT_DIR`, `NUM_GPUS`, `STAGE1_CKPT`, and API tokens through the environment or Kubernetes Secrets instead of hardcoding them.
