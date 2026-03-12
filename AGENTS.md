# AGENTS.md
## Purpose
- This repo is a Python / PyTorch research codebase for RF fingerprinting and domain generalization.
- Primary entrypoint: `main.py`. Supporting modules live in `util/`.
- No prior `AGENTS.md`, `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` were found during this scan.
## Repository Map
- `main.py`: CLI entrypoint for `train`, `test`, and `train_test`.
- `util/CNNmodel.py`: `MACNN` architecture and attention blocks.
- `util/augmentation.py`: NumPy augmentation functions.
- `util/con_losses.py`: supervised contrastive loss plus a smoke test.
- `util/get_dataset.py`: dataset loading for `ORACLE` and `WiSig`.
- `Dataset_ORALCE/`: ORACLE data. Keep the misspelling; the code depends on it.
- `Dataset_WiSig/`: WiSig pickle files.
- `weight/`: saved checkpoints.
- `log/`: logs / outputs.
- `opencode.json`: sensitive local config; do not expose or commit its secrets.
## Environment
- Language: Python.
- Inferred dependencies: `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `torchsummary`.
- No `requirements.txt`, `pyproject.toml`, or package manager config exists.
- No formal build system exists; treat this as a script-driven project.
- CUDA is optional and controlled with `--cuda` plus `torch.cuda.is_available()`.
## Setup
- Create a Python environment manually.
- Minimum practical install set:
```bash
pip install torch torchvision numpy scipy scikit-learn torchsummary
```
## Build / Sanity Commands
- There is no real build step.
- Fast syntax check:
```bash
python -m py_compile main.py util/*.py
```
- Broader compile pass:
```bash
python -m compileall main.py util
```
## Lint / Format
- No linter or formatter is configured in-repo.
- Do not assume `ruff`, `black`, `flake8`, or `isort` are installed.
- Use `python -m py_compile main.py util/*.py` as the default pre-change verification.
- If you use a local formatter, keep edits tightly scoped; do not reformat the whole repo.
## Test Commands
- There is no formal `tests/` directory and no `pytest` config.
- Existing verification is smoke-test oriented.
- Fastest module smoke test:
```bash
python util/con_losses.py
```
- CLI sanity check:
```bash
python main.py --help
```
- Syntax validation after edits:
```bash
python -m py_compile main.py util/*.py
```
## Running A Single Test
- There is no built-in single-test runner right now.
- Best current narrow check:
```bash
python util/con_losses.py
```
- For a narrow CLI path, prefer `python main.py --help` or a targeted `main.py` invocation instead of full training.
- If the repo later gains `pytest`, use:
```bash
python -m pytest tests/test_file.py::test_name -q
```
## Train / Eval Commands
- Train only:
```bash
python main.py --mode train --dataset_name ORACLE --model_size S
```
- Train and test:
```bash
python main.py --mode train_test --dataset_name WiSig --model_size S
```
- Test only:
```bash
python main.py --mode test --dataset_name ORACLE --model_size S
```
- Test-only runs require a checkpoint whose filename matches the CLI arguments.
## Known Runtime Gotchas
- `main.py` saves and loads full model objects, not `state_dict`s.
- In PyTorch 2.6+, `torch.load()` defaults to `weights_only=True`; legacy checkpoints in `weight/` can fail to load unless you explicitly opt out or allowlist globals.
- Only use `weights_only=False` for trusted checkpoints.
- Do not casually change checkpoint serialization; it affects compatibility with existing files.
- `util/get_dataset.py` has a `__main__` block, but it appears stale and references mismatched names; do not rely on it as a regression test.
- Preserve the `Dataset_ORALCE` spelling unless you update every related path.
## Code Style
### General
- Follow existing Python + PyTorch conventions, but prefer small, local cleanups over broad rewrites.
- Use 4-space indentation.
- Keep files ASCII unless a file already needs non-ASCII text.
- Prefer clear, direct code over clever abstractions.
- Keep helpers focused and side effects obvious.
### Imports
- Order imports as: standard library, third-party, local modules.
- Separate import groups with one blank line.
- Prefer explicit imports in new code.
- `main.py` currently uses `from util.CNNmodel import *`; do not introduce more wildcard imports.
- Only clean import blocks in files you already touch.
### Formatting
- Match the surrounding file style first.
- Keep multiline calls and literals vertically readable.
- Use trailing commas in multiline structures when they make diffs cleaner.
- Avoid decorative comments, banner comments, and unnecessary whitespace churn.
### Types
- Existing typing is light.
- Add type hints for new public helpers and non-obvious return values when practical.
- Do not turn a small fix into a full typing refactor.
- Document important tensor / ndarray shape expectations in code or docstrings when needed.
### Naming
- Use `snake_case` for functions, variables, and helpers.
- Use `PascalCase` for classes.
- Use `UPPER_CASE` for constants.
- Prefer short, domain-specific names like `train_loader`, `val_loss`, and `num_classes`.
- Avoid new abbreviations unless they are already standard in the file.
### Data Handling
- Be careful with shapes; this code often uses transpose, squeeze, and channel-first tensors.
- Preserve `Conv1d` channel-first assumptions.
- Match current tensor conversion patterns when moving data from NumPy to PyTorch.
- Keep CPU / CUDA handling explicit.
- Do not silently change dtypes in preprocessing or augmentation code.
### Error Handling
- Validate user-facing config early.
- Raise explicit exceptions such as `ValueError` for unsupported options.
- Include the invalid value in failure messages when possible.
- Do not swallow dataset-loading or checkpoint-loading errors.
- Keep fallback behavior explicit and easy to remove.
### CLI / Config Changes
- Preserve the existing `argparse` interface unless the task truly requires a breaking change.
- Prefer adding a new flag over changing the meaning of an existing one.
- If you add a flag, update checkpoint naming logic if it affects saved model identity.
- Keep defaults conservative; expensive runs should stay explicit.
### Model / Training Changes
- Keep model code in `util/CNNmodel.py`, losses in `util/con_losses.py`, data logic in `util/get_dataset.py`, and orchestration in `main.py`.
- Keep training, validation, and test loops straightforward.
- Maintain deterministic seed behavior unless the task explicitly changes reproducibility policy.
- Call out checkpoint-compatibility changes in your final report.
## Validation Expectations For Agents
- After a small edit, run the narrowest relevant check first.
- Default order:
  1. `python -m py_compile main.py util/*.py`
  2. `python util/con_losses.py` if loss code changed
  3. `python main.py --help` if CLI code changed
- Avoid full training unless the task requires it.
- If a command depends on large datasets or trusted checkpoints, say so clearly.
## Security
- Treat `opencode.json` as sensitive.
- Do not commit secrets, tokens, or machine-specific paths.
- Be careful with trusted-model loading when using `torch.load(..., weights_only=False)`.
- Avoid destructive edits to datasets, checkpoints, or logs.
## What To Include In Change Summaries
- Which files changed.
- Whether the change affects training, evaluation, dataset loading, augmentation, or checkpoint compatibility.
- Which validation command you ran.
- Any dataset or checkpoint assumptions needed to reproduce the result.
