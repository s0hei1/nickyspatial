# -*- coding: utf-8 -*-
"""Helpers , Aren't they useful ?"""

import json
import os

import numpy as np


def create_sample_data():
    """Create a synthetic 4-band (B, G, R, NIR) image for testing.

    Returns:
    --------
    image_data : numpy.ndarray
        Synthetic image data
    transform : affine.Affine
        Affine transformation for the raster
    crs : rasterio.crs.CRS
        Coordinate reference system
    """
    ## need to write logic for this one to create a pseduo image

    # return image_data, transform, crs
    return None


def calculate_statistics_summary(layer_manager, output_file=None):
    """Calculate summary statistics for all layers in a layer manager.

    Parameters:
    -----------
    layer_manager : LayerManager
        Layer manager containing layers
    output_file : str, optional
        Path to save the summary to (as JSON)

    Returns:
    --------
    summary : dict
        Dictionary with summary statistics
    """
    summary = {}

    for layer_name in layer_manager.get_layer_names():
        layer = layer_manager.get_layer(layer_name)

        layer_summary = {
            "type": layer.type,
            "created_at": str(layer.created_at),
            "parent": layer.parent.name if layer.parent else None,
        }

        if layer.objects is not None:
            layer_summary["object_count"] = len(layer.objects)

            if "area_units" in layer.objects.columns:
                layer_summary["total_area"] = float(layer.objects["area_units"].sum())
                layer_summary["mean_area"] = float(layer.objects["area_units"].mean())

            for col in layer.objects.columns:
                if col.lower().endswith("class") or col.lower() == "classification":
                    class_counts = layer.objects[col].value_counts().to_dict()
                    layer_summary[f"{col}_counts"] = {str(k): int(v) for k, v in class_counts.items() if k is not None}

        if layer.attached_functions:
            layer_summary["functions"] = list(layer.attached_functions.keys())

        summary[layer_name] = layer_summary

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def get_band_statistics(image_data, band_names=None):
    """Calculate statistics for each band in a raster image.

    Parameters:
    -----------
    image_data : numpy.ndarray
        Raster image data (bands, height, width)
    band_names : list of str, optional
        Names of the bands

    Returns:
    --------
    stats : dict
        Dictionary with band statistics
    """
    num_bands = image_data.shape[0]

    if band_names is None:
        band_names = [f"Band_{i + 1}" for i in range(num_bands)]

    stats = {}

    for i, band_name in enumerate(band_names):
        if i >= num_bands:
            break

        band_data = image_data[i]
        stats[band_name] = {
            "min": float(np.min(band_data)),
            "max": float(np.max(band_data)),
            "mean": float(np.mean(band_data)),
            "std": float(np.std(band_data)),
            "median": float(np.median(band_data)),
            "percentile_5": float(np.percentile(band_data, 5)),
            "percentile_95": float(np.percentile(band_data, 95)),
        }

    return stats


def memory_usage(layer):
    """Estimate memory usage of a layer in MB.

    Parameters:
    -----------
    layer : Layer
        Layer to calculate memory usage for

    Returns:
    --------
    memory_mb : float
        Estimated memory usage in MB
    """
    import sys

    memory = 0

    if layer.raster is not None:
        memory += layer.raster.nbytes

    if layer.objects is not None:
        for col in layer.objects.columns:
            if col != "geometry":
                memory += sys.getsizeof(layer.objects[col].values)

        memory += len(layer.objects) * 1000
    memory_mb = memory / (1024 * 1024)

    return memory_mb
