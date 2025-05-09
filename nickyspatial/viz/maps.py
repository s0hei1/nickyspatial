# -*- coding: utf-8 -*-
"""Functions to create maps and visualize layers."""

import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_layer(
    layer,
    image_data=None,
    attribute=None,
    title=None,
    rgb_bands=(2, 1, 0),
    figsize=(12, 10),
    cmap="viridis",
    show_boundaries=False,
):
    """Plot a layer, optionally with an attribute or image backdrop."""
    fig, ax = plt.subplots(figsize=figsize)

    if title:
        ax.set_title(title)
    elif attribute:
        ax.set_title(f"{attribute} by Segment")
    else:
        ax.set_title("Layer Visualization")

    if image_data is not None:
        num_bands = image_data.shape[0]
        if num_bands >= 3 and max(rgb_bands) < num_bands:
            r = image_data[rgb_bands[0]]
            g = image_data[rgb_bands[1]]
            b = image_data[rgb_bands[2]]

            r_norm = np.clip((r - r.min()) / (r.max() - r.min() + 1e-10), 0, 1)
            g_norm = np.clip((g - g.min()) / (g.max() - g.min() + 1e-10), 0, 1)
            b_norm = np.clip((b - b.min()) / (b.max() - b.min() + 1e-10), 0, 1)

            rgb = np.stack([r_norm, g_norm, b_norm], axis=2)

            ax.imshow(rgb)
        else:
            gray = image_data[0]
            gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
            ax.imshow(gray_norm, cmap="gray")

    if attribute and attribute in layer.objects.columns:
        layer.objects.plot(
            column=attribute,
            cmap=cmap,
            ax=ax,
            legend=True,
            alpha=0.7 if image_data is not None else 1.0,
        )

    if show_boundaries and layer.raster is not None:
        from skimage.segmentation import mark_boundaries

        if image_data is not None:
            if "num_bands" in locals() and num_bands >= 3:
                base_img = rgb
            else:
                gray = image_data[0]
                gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
                base_img = np.stack([gray_norm, gray_norm, gray_norm], axis=2)

            bounded = mark_boundaries(base_img, layer.raster, color=(1, 1, 0), mode="thick")

            if attribute is None:
                ax.imshow(bounded)
        else:
            ax.imshow(
                mark_boundaries(
                    np.zeros((layer.raster.shape[0], layer.raster.shape[1], 3)),
                    layer.raster,
                    color=(1, 1, 0),
                    mode="thick",
                )
            )

    ax.grid(alpha=0.3)
    return fig


# def plot_classification(layer, class_field="classification", figsize=(12, 10), legend=True, classes=[]):
#     """Plot classified segments with different colors for each class."""
#     fig, ax = plt.subplots(figsize=figsize)

#     if class_field not in layer.objects.columns:
#         raise ValueError(f"Class field '{class_field}' not found in layer objects")

#     class_values = [v for v in layer.objects[class_field].unique() if v is not None]

#     num_classes = len(class_values)

#     colors = plt.cm.tab20(np.linspace(0, 1, max(num_classes, 1)))

#     cmap = ListedColormap(colors)
#     class_map = {value: i for i, value in enumerate(class_values)}
#     layer.objects["_class_id"] = layer.objects[class_field].map(class_map)

#     layer.objects.plot(
#         column="_class_id",
#         cmap=cmap,
#         ax=ax,
#         edgecolor="black",
#         linewidth=0.5,
#         legend=False,
#     )

#     if legend and len(class_values) > 0:
#         patches = [mpatches.Patch(color=colors[i], label=value) for i, value in enumerate(class_values)]
#         ax.legend(handles=patches, loc="upper right", title=class_field)

#     ax.set_title(f"Classification by {class_field}")
#     ax.set_xlabel("X Coordinate")
#     ax.set_ylabel("Y Coordinate")

#     if "_class_id" in layer.objects.columns:
#         layer.objects = layer.objects.drop(columns=["_class_id"])

#     return fig


def plot_classification(layer, class_field="classification", figsize=(12, 10), legend=True, classes=None):
    """Plot classified segments with different colors for each class."""
    fig, ax = plt.subplots(figsize=figsize)

    if class_field not in layer.objects.columns:
        raise ValueError(f"Class field '{class_field}' not found in layer objects")

    class_values = [v for v in layer.objects[class_field].unique() if v is not None]

    # generate base colormap
    base_colors = plt.cm.tab20(np.linspace(0, 1, max(len(class_values), 1)))

    colors_list = []
    for idx, class_value in enumerate(class_values):
        if classes and class_value in list(classes.keys()):
            # reuse stored color
            color_hex = classes[class_value]["color"]
        else:
            # assign new color (from tab20 or random if exceeds)
            if idx < len(base_colors):
                rgb = base_colors[idx][:3]
                color_hex = "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            else:
                color_hex = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            classes[class_value] = {"color": color_hex, "sample_ids": []}

        # convert hex → RGB tuple for ListedColormap
        rgb_tuple = tuple(int(color_hex[i : i + 2], 16) / 255 for i in (1, 3, 5))
        colors_list.append(rgb_tuple)

    # create colormap
    cmap = ListedColormap(colors_list)

    # map class values to indices
    class_map = {value: i for i, value in enumerate(class_values)}
    layer.objects["_class_id"] = layer.objects[class_field].map(class_map)

    layer.objects.plot(
        column="_class_id",
        cmap=cmap,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    if legend and len(class_values) > 0:
        patches = [mpatches.Patch(color=classes[value]["color"], label=value) for value in class_values]
        ax.legend(handles=patches, loc="upper right", title=class_field)

    ax.set_title(f"Classification by {class_field}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # cleanup temporary column
    if "_class_id" in layer.objects.columns:
        layer.objects = layer.objects.drop(columns=["_class_id"])

    return fig


def plot_comparison(
    before_layer,
    after_layer,
    attribute=None,
    class_field=None,
    figsize=(16, 8),
    title=None,
):
    """Plot before and after views of layers for comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if title:
        fig.suptitle(title)

    if attribute and attribute in before_layer.objects.columns:
        before_layer.objects.plot(column=attribute, ax=ax1, legend=True)
        ax1.set_title(f"Before: {attribute}")
    elif class_field and class_field in before_layer.objects.columns:
        class_values = [v for v in before_layer.objects[class_field].unique() if v is not None]
        num_classes = len(class_values)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_classes, 1)))
        cmap = ListedColormap(colors)
        class_map = {value: i for i, value in enumerate(class_values)}
        before_layer.objects["_class_id"] = before_layer.objects[class_field].map(class_map)

        before_layer.objects.plot(
            column="_class_id",
            cmap=cmap,
            ax=ax1,
            edgecolor="black",
            linewidth=0.5,
            legend=False,
        )

        patches = [mpatches.Patch(color=colors[i], label=value) for i, value in enumerate(class_values)]
        ax1.legend(handles=patches, loc="upper right", title=class_field)
        ax1.set_title(f"Before: {class_field}")
    else:
        before_layer.objects.plot(ax=ax1)
        ax1.set_title("Before")

    if attribute and attribute in after_layer.objects.columns:
        after_layer.objects.plot(column=attribute, ax=ax2, legend=True)
        ax2.set_title(f"After: {attribute}")
    elif class_field and class_field in after_layer.objects.columns:
        class_values = [v for v in after_layer.objects[class_field].unique() if v is not None]
        num_classes = len(class_values)
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_classes, 1)))
        cmap = ListedColormap(colors)
        class_map = {value: i for i, value in enumerate(class_values)}
        after_layer.objects["_class_id"] = after_layer.objects[class_field].map(class_map)

        after_layer.objects.plot(
            column="_class_id",
            cmap=cmap,
            ax=ax2,
            edgecolor="black",
            linewidth=0.5,
            legend=False,
        )

        patches = [mpatches.Patch(color=colors[i], label=value) for i, value in enumerate(class_values)]
        ax2.legend(handles=patches, loc="upper right", title=class_field)
        ax2.set_title(f"After: {class_field}")
    else:
        after_layer.objects.plot(ax=ax2)
        ax2.set_title("After")

    if "_class_id" in before_layer.objects.columns:
        before_layer.objects = before_layer.objects.drop(columns=["_class_id"])
    if "_class_id" in after_layer.objects.columns:
        after_layer.objects = after_layer.objects.drop(columns=["_class_id"])

    return fig
