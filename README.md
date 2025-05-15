# NickySpatial

An open-source object-based image analysis library for remote sensing.

| Category | Badge |
|----------|-------|
| **Build** | [![uv](https://img.shields.io/badge/build-uv-blue?logo=uv&logoColor=white)](https://pypi.org/project/uv/) [![hatchling](https://img.shields.io/badge/build-hatchling-blue?logo=hatchling&logoColor=white)](https://github.com/pypa/hatchling)  [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kshitijrajsharma/nickyspatial/master.svg)](https://results.pre-commit.ci/latest/github/kshitijrajsharma/nickyspatial/master)  |
| **Code Quality** | [![Ruff](https://img.shields.io/badge/ruff-v0.11.3-blue?logo=ruff&logoColor=white)](https://beta.ruff.rs/) [![pre-commit](https://img.shields.io/badge/pre--commit-active-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com) ![Tests](https://img.shields.io/badge/tests-passing-brightgreen) ![Coverage](https://img.shields.io/badge/Coverage-90%25-brightgreen) |
| **Documentation** | [![MkDocs](https://img.shields.io/badge/MkDocs-Documentation-blue?logo=mkdocs&logoColor=white)](https://kshitijrajsharma.github.io/nickyspatial/) [![Read DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/kshitijrajsharma/nickyspatial) |
| **Package Info** | ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) [![PyPI version](https://img.shields.io/pypi/v/nickyspatial.svg)](https://pypi.org/project/nickyspatial) ![Dependencies](https://img.shields.io/librariesio/github/kshitijrajsharma/nickyspatial) ![License](https://img.shields.io/badge/License-MIT-yellow.svg) |
| **Community** | ![GitHub Stars](https://img.shields.io/github/stars/kshitijrajsharma/nickyspatial?style=social) ![Issues](https://img.shields.io/github/issues/kshitijrajsharma/nickyspatial) ![Downloads](https://img.shields.io/pypi/dm/nickyspatial.svg) ![Last Commit](https://img.shields.io/github/last-commit/kshitijrajsharma/nickyspatial) |

> [!WARNING]
> This project is under active development and lot of its functionality is still in my head yet to code.

Find Demo Frontend Here : https://nickyspatial-gpoqz.ondigitalocean.app/ 

## Description

NickySpatial is a Python package that provides object-based image analysis (OBIA) functionality similar to commercial software like eCognition. It allows users to segment geospatial imagery into meaningful objects, calculate statistics, and apply rule-based classification.

## Project Structure

```graphql
nickyspatial/
├── __init__.py
├── io/
│   ├── __init__.py
│   ├── raster.py       # Raster I/O
│   └── vector.py       # Vector I/O
├── core/
│   ├── __init__.py
│   ├── layer.py        # Layer class and management
│   ├── segmentation.py # Segmentation algorithms
│   └── rules.py        # Rule engine
├── stats/
│   ├── __init__.py
│   ├── basic.py        # Basic statistics (min, max, mean, etc.)
│   ├── spatial.py      # Spatial statistics (area, perimeter, etc.)
│   └── spectral.py     # Spectral indices (NDVI, etc.)
├── filters/
│   ├── __init__.py
│   ├── spatial.py      # Spatial filters (smoothing, merging)
│   └── spectral.py     # Spectral filters (band math)
├── viz/
│   ├── __init__.py
│   ├── maps.py         # Map visualization
│   └── charts.py       # Statistical charts
└── utils/
    ├── __init__.py
    └── helpers.py      # Helper functions
```

## Installation

```bash
pip install nickyspatial
```

## Quick Start

```python
import nickyspatial as ns
 TODO : add sample computation here

```

## Documentation

For detailed documentation and examples, see the [documentation website](https://kshitijrajsharma.github.io/nickyspatial/).
Deepwiki also provides documentation quite detailed , Do check it [out](https://deepwiki.com/kshitijrajsharma/nickyspatial ) if you like

## Examples

Check out the 'examples' directory for more examples:

TODO : Add example scripts here

## Contributing

Contributions are welcome! Follow [dev setup guide](./docs/dev.md) & Please feel free to submit a Pull Request.

## Acknowledgments

- Inspired by the functionality of eCognition and other OBIA methodologies
- Built on top of powerful open-source libraries like numpy, rasterio, scikit-image, and GeoPandas
- **Nicky**  : Nicky is my belated dog and I named this package in his memory!
    <p align="left">
      <img src="https://github.com/user-attachments/assets/b5b86c63-ae5a-48b4-9d45-3bb34a58a102" alt="Nicky the dog" width="160" style="border-radius: 80px;" />
    </p>


## Contributors

[![Contributors](https://contrib.rocks/image?repo=kshitijrajsharma/nickyspatial)](https://github.com/kshitijrajsharma/nickyspatial/graphs/contributors)
