## Development Setup 

- Install uv 

UV can be installed in many ways , Follow [official docs](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) to install based on your OS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Install this project dependencies to `.venv` directly 

```bash
uv sync
``` 

- Install  `pre-commit` hooks

```bash
pre-commit install
```