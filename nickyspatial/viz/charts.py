# -*- coding: utf-8 -*-
"""Visualization functions for plotting histograms, statistics, and scatter plots."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_histogram(layer, attribute, bins=20, figsize=(10, 6), by_class=None):
    """Plot a histogram of attribute values.

    Parameters:
    -----------
    layer : Layer
        Layer containing data
    attribute : str
        Attribute to plot
    bins : int
        Number of bins
    figsize : tuple
        Figure size
    by_class : str, optional
        Column to group by (e.g., 'classification')

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if layer.objects is None or attribute not in layer.objects.columns:
        raise ValueError(f"Attribute '{attribute}' not found in layer objects")

    fig, ax = plt.subplots(figsize=figsize)

    if by_class and by_class in layer.objects.columns:
        data = layer.objects[[attribute, by_class]].copy()

        for class_value, group in data.groupby(by_class):
            if class_value is None:
                continue

            sns.histplot(group[attribute], bins=bins, alpha=0.6, label=str(class_value), ax=ax)

        ax.legend(title=by_class)
    else:
        sns.histplot(layer.objects[attribute], bins=bins, ax=ax)

    ax.set_title(f"Histogram of {attribute}")
    ax.set_xlabel(attribute)
    ax.set_ylabel("Count")

    return fig


def plot_statistics(layer, stats_dict, figsize=(12, 8), kind="bar", y_log=False):
    """Plot statistics from a statistics dictionary.

    Parameters:
    -----------
    layer : Layer
        Layer the statistics are calculated for
    stats_dict : dict
        Dictionary with statistics (from attach_* functions)
    figsize : tuple
        Figure size
    kind : str
        Plot type: 'bar', 'line', or 'pie'
    y_log : bool
        Whether to use logarithmic scale for y-axis

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    flat_stats = {}

    def _flatten_dict(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                _flatten_dict(value, f"{prefix}{key}_")
            else:
                flat_stats[f"{prefix}{key}"] = value

    _flatten_dict(stats_dict)

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "pie" and "class_percentages" in stats_dict:
        percentages = stats_dict["class_percentages"]
        values = list(percentages.values())
        labels = list(percentages.keys())

        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
        ax.axis("equal")
        ax.set_title("Class Distribution")

    elif kind == "pie" and "percentages" in flat_stats:
        percentages = pd.Series(flat_stats).filter(like="percentage")
        values = percentages.values
        labels = [label.replace("_percentage", "") for label in percentages.index]

        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, shadow=True)
        ax.axis("equal")
        ax.set_title("Distribution")

    else:
        stats_df = pd.DataFrame({"Metric": list(flat_stats.keys()), "Value": list(flat_stats.values())})

        if kind != "line":
            stats_df = stats_df.sort_values("Value", ascending=False)

        if kind == "bar":
            sns.barplot(x="Metric", y="Value", data=stats_df, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        elif kind == "line":
            sns.lineplot(x="Metric", y="Value", data=stats_df, ax=ax, marker="o")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        if y_log:
            ax.set_yscale("log")

        ax.set_title("Statistics Summary")

    plt.tight_layout()
    return fig


def plot_scatter(layer, x_attribute, y_attribute, color_by=None, figsize=(10, 8)):
    """Create a scatter plot of two attributes.

    Parameters:
    -----------
    layer : Layer
        Layer containing data
    x_attribute : str
        Attribute for x-axis
    y_attribute : str
        Attribute for y-axis
    color_by : str, optional
        Attribute to color points by
    figsize : tuple
        Figure size

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if layer.objects is None or x_attribute not in layer.objects.columns or y_attribute not in layer.objects.columns:
        raise ValueError("Attributes not found in layer objects")

    fig, ax = plt.subplots(figsize=figsize)

    if color_by and color_by in layer.objects.columns:
        scatter = ax.scatter(
            layer.objects[x_attribute],
            layer.objects[y_attribute],
            c=layer.objects[color_by],
            cmap="viridis",
            alpha=0.7,
            s=50,
            edgecolor="k",
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        ax.scatter(
            layer.objects[x_attribute],
            layer.objects[y_attribute],
            alpha=0.7,
            s=50,
            edgecolor="k",
        )

    ax.set_title(f"{y_attribute} vs {x_attribute}")
    ax.set_xlabel(x_attribute)
    ax.set_ylabel(y_attribute)
    ax.grid(alpha=0.3)

    return fig
