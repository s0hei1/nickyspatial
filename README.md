# NickySpatial

An open-source object-based image analysis library for remote sensing. 

| **Build System**      | [![uv](https://img.shields.io/badge/build-uv-blue?logo=uv&logoColor=white)](https://pypi.org/project/uv/) [![hatchling](https://img.shields.io/badge/build-hatchling-blue?logo=hatchling&logoColor=white)](https://github.com/pypa/hatchling) |
| **Linter & Formatter**| [![Ruff](https://img.shields.io/badge/ruff-v0.0.0-blue?logo=ruff&logoColor=white)](https://beta.ruff.rs/) via [![pre-commit](https://img.shields.io/badge/pre--commit-active-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com) |
| **Documentation**     | Built with [![MkDocs](https://img.shields.io/badge/MkDocs-Documentation-blue?logo=mkdocs&logoColor=white)](https://www.mkdocs.org/) – [View Docs](https://kshitijrajsharma.github.io/nickyspatial/) |
| **Tests**             | ![Tests](https://img.shields.io/badge/tests-passing-brightgreen) |
| **Coverage**          | ![Coverage](https://img.shields.io/badge/Coverage-90%25-brightgreen) |
| **Dependencies**      | ![Dependencies](https://img.shields.io/librariesio/github/kshitijrajsharma/nickyspatial) |
| **Python Version**    | ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) |
| **PyPI Version**      | [![PyPI version](https://img.shields.io/pypi/v/nickyspatial.svg)](https://pypi.org/project/nickyspatial) |
| **Downloads**         | ![Downloads](https://img.shields.io/pypi/dm/nickyspatial.svg) |
| **License**           | ![License](https://img.shields.io/badge/License-MIT-yellow.svg) |
| **GitHub Stars**      | ![GitHub Stars](https://img.shields.io/github/stars/kshitijrajsharma/nickyspatial?style=social) |
| **Issues**            | ![Issues](https://img.shields.io/github/issues/kshitijrajsharma/nickyspatial) |
| **Latest Commit**     | [View Commit](https://github.com/kshitijrajsharma/nickyspatial/commits/master) |
| **Donate**            | [![Donate](https://img.shields.io/badge/Donate-PayPal-blue)](https://www.paypal.me/yourlink) |
> [!WARNING]
> This project is under active development and lot of its functionality is still in my head yet to code.

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

For detailed documentation and examples, see the [documentation website](#).

## Examples

Check out the [examples](./docs/examples/) directory for more examples:
 

TODO : Add example scripts here 

## Contributing

Contributions are welcome! Follow [dev setup guide](./docs/dev.md) & Please feel free to submit a Pull Request.

## Acknowledgments

- Inspired by the functionality of eCognition and other OBIA methodologies
- Built on top of powerful open-source libraries like numpy, rasterio, scikit-image, and GeoPandas

### Nicky
**Nicky** is my belated dog and I named this package on his memory ! 

![image](https://github.com/user-attachments/assets/b5b86c63-ae5a-48b4-9d45-3bb34a58a102)
