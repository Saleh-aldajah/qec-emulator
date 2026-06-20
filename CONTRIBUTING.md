# Contributing to QEC Emulator

Thank you for your interest in contributing. This project welcomes bug reports,
feature requests, documentation improvements, and code contributions.

## Getting help and reporting issues

- **Questions:** open a GitHub issue with the `question` label.
- **Bug reports:** open an issue and include the QEC Emulator version
  (`qec-emulator info`), your Python version and OS, the exact command that
  failed, the full traceback, and a minimal reproducible example.
- **Feature requests:** open an issue with the `enhancement` label describing
  the use case and desired behaviour.

## Development setup

```bash
git clone https://github.com/Saleh-aldajah/qec-emulator.git
cd qec-emulator
pip install .
pip install matplotlib fastapi typer httpx uvicorn
```

## Running the tests

```bash
pytest          # expects 67 passed, 3 skipped
```

New functionality should be accompanied by tests. Bug fixes should include a
regression test that fails before the fix and passes after.

## Reproducibility expectations

Any change that touches code construction, decoders, noise models, or benchmark
drivers must preserve SHA-256 matrix pinning and deterministic context-derived
seeding, and must be validated by re-running `reproduce_all.py` and
`verify_hashes.py`. Note in your pull request if any archived result changes.

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with tests and documentation.
3. Ensure `pytest` is green and CI passes.
4. Open a pull request linking any related issue.

By contributing you agree your contributions will be licensed under the MIT License.