# NickySpatial

An open-source object-based image analysis library for remote sensing. 

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

This project is follows [uv](https://docs.astral.sh/uv/) for dependencies management and project publishing

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

Check out the examples directory for more detailed examples:
 

TODO : Add example scripts here 

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the functionality of eCognition and other OBIA methodologies
- Built on top of powerful open-source libraries like numpy, rasterio, scikit-image, and GeoPandas

#### **Nicky** is my belated dog and I named this package on his memory ! 

![image](https://github.com/user-attachments/assets/b5b86c63-ae5a-48b4-9d45-3bb34a58a102)
