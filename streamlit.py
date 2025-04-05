# -*- coding: utf-8 -*-
"""Streamlit Webapp

Frontend for the demo of nickyspatial library
"""

import datetime
import os
import tempfile

import pandas as pd

import streamlit as st
from nickyspatial import (
    LayerManager,
    MultiResolutionSegmentation,
    RuleSet,
    attach_area_stats,
    attach_ndvi,
    attach_shape_metrics,
    attach_spectral_indices,
    layer_to_raster,
    layer_to_vector,
    plot_classification,
    plot_layer,
    read_raster,
)

st.set_page_config(page_title="nickyspatial - Remote Sensing Analysis", page_icon="🛰️", layout="wide")


@st.cache_resource
def get_temp_dir():
    """Get a temporary directory for storing output files."""
    return tempfile.mkdtemp()


def initialize_session_state():
    """Initialize session state variables."""
    if "manager" not in st.session_state:
        st.session_state.manager = LayerManager()
    if "image_data" not in st.session_state:
        st.session_state.image_data = None
    if "transform" not in st.session_state:
        st.session_state.transform = None
    if "crs" not in st.session_state:
        st.session_state.crs = None
    if "layers" not in st.session_state:
        st.session_state.layers = {}
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = get_temp_dir()
    if "rule_sets" not in st.session_state:
        st.session_state.rule_sets = {}
    if "active_ruleset" not in st.session_state:
        st.session_state.active_ruleset = None
    if "band_mappings" not in st.session_state:
        st.session_state.band_mappings = {}
    if "available_attributes" not in st.session_state:
        st.session_state.available_attributes = set()


def load_raster(file_path):
    """Load raster data and initialize session state variables."""
    try:
        with st.spinner("Reading raster data..."):
            image_data, transform, crs = read_raster(file_path)
            st.session_state.image_data = image_data
            st.session_state.transform = transform
            st.session_state.crs = crs
            st.session_state.manager = LayerManager()
            st.session_state.layers = {}
            st.session_state.rule_sets = {}

            num_bands = image_data.shape[0]
            default_mappings = {}
            if num_bands >= 4:
                default_mappings = {"blue": "band_1_mean", "green": "band_2_mean", "red": "band_3_mean", "nir": "band_4_mean"}
            elif num_bands == 3:
                default_mappings = {"blue": "band_1_mean", "green": "band_2_mean", "red": "band_3_mean"}

            st.session_state.band_mappings = default_mappings
            update_available_attributes()

        return True
    except Exception as e:
        st.error(f"Error loading raster data: {str(e)}")
        return False


def get_layer_attributes(layer_name):
    """Get available attributes from the specified layer."""
    if layer_name and layer_name in st.session_state.layers:
        layer = st.session_state.layers[layer_name]
        if layer and hasattr(layer, "objects") and hasattr(layer.objects, "columns"):
            return [col for col in layer.objects.columns if col not in ["geometry"]]
    return []


def update_available_attributes():
    """Update the available attributes based on the current layers."""
    attributes = set()
    for layer_name, layer in st.session_state.layers.items():
        if layer and hasattr(layer, "objects") and hasattr(layer.objects, "columns"):
            for col in layer.objects.columns:
                if col not in ["geometry"]:
                    attributes.add(col)
    st.session_state.available_attributes = attributes


def perform_segmentation(image_data, transform, crs, scale_param, compactness_param, segmentation_name):
    """Perform segmentation on the image data."""
    try:
        with st.spinner("Performing segmentation..."):
            segmenter = MultiResolutionSegmentation(
                scale=scale_param,
                compactness=compactness_param,
            )

            segmentation_layer = segmenter.execute(
                image_data,
                transform,
                crs,
                layer_manager=st.session_state.manager,
                layer_name=segmentation_name,
            )

            st.session_state.layers[segmentation_name] = segmentation_layer
            update_available_attributes()
            return segmentation_layer
    except Exception as e:
        st.error(f"Error during segmentation: {str(e)}")
        return None


def calculate_ndvi(layer, nir_column, red_column, output_column="NDVI"):
    """Calculate NDVI for the specified layer."""
    try:
        with st.spinner("Calculating NDVI..."):
            layer.attach_function(
                attach_ndvi,
                name="ndvi_stats",
                nir_column=nir_column,
                red_column=red_column,
                output_column=output_column,
            )
            update_available_attributes()
        return True
    except Exception as e:
        st.error(f"Error calculating NDVI: {str(e)}")
        return False


def calculate_spectral_indices(layer, band_mappings):
    """Calculate spectral indices for the specified layer."""
    try:
        with st.spinner("Calculating spectral indices..."):
            layer.attach_function(attach_spectral_indices, name="spectral_indices", bands=band_mappings)
            update_available_attributes()
        return True
    except Exception as e:
        st.error(f"Error calculating spectral indices: {str(e)}")
        return False


def calculate_shape_metrics(layer):
    """Calculate shape metrics for the specified layer."""
    try:
        with st.spinner("Calculating shape metrics..."):
            layer.attach_function(attach_shape_metrics, name="shape_metrics")
            update_available_attributes()
        return True
    except Exception as e:
        st.error(f"Error calculating shape metrics: {str(e)}")
        return False


def apply_rule_set(ruleset, input_layer, output_layer_name, result_field):
    """Apply the specified rule set to the input layer."""
    try:
        with st.spinner(f"Applying {ruleset.name} rules..."):
            result_layer = ruleset.execute(
                input_layer, layer_manager=st.session_state.manager, layer_name=output_layer_name, result_field=result_field
            )

            st.session_state.layers[output_layer_name] = result_layer
            update_available_attributes()
            return result_layer
    except Exception as e:
        st.error(f"Error applying rule set: {str(e)}")
        return None


def calculate_area_stats(layer, class_field):
    """Calculate area statistics for the specified layer."""
    try:
        with st.spinner("Calculating area statistics..."):
            layer.attach_function(attach_area_stats, name="area_by_class", by_class=class_field)
            return layer.get_function_result("area_by_class")
    except Exception as e:
        st.error(f"Error calculating area statistics: {str(e)}")
        return None


def export_vector(layer, export_filepath):
    """Export the specified layer as a GeoJSON file."""
    try:
        with st.spinner("Exporting as GeoJSON..."):
            layer_to_vector(layer, export_filepath)
            return True
    except Exception as e:
        st.error(f"Error exporting vector: {str(e)}")
        return False


def export_raster(layer, export_filepath, column):
    """Export the specified layer as a GeoTIFF file."""
    try:
        with st.spinner("Exporting as GeoTIFF..."):
            layer_to_raster(layer, export_filepath, column=column)
            return True
    except Exception as e:
        st.error(f"Error exporting raster: {str(e)}")
        return False


def create_example_rule_sets():
    """Create example rule sets for demonstration."""
    try:
        with st.spinner("Creating example rule sets..."):
            land_cover_rules = RuleSet(name="Land_Cover")
            land_cover_rules.add_rule(name="Vegetation", condition="NDVI > 0.2")
            land_cover_rules.add_rule(name="Other", condition="NDVI <= 0.2")

            vegetation_rules = RuleSet(name="Vegetation_Types")
            vegetation_rules.add_rule(name="Healthy_Vegetation", condition="(classification == 'Vegetation') & (NDVI > 0.6)")
            vegetation_rules.add_rule(
                name="Moderate_Vegetation", condition="(classification == 'Vegetation') & (NDVI <= 0.6) & (NDVI > 0.4)"
            )
            vegetation_rules.add_rule(name="Sparse_Vegetation", condition="(classification == 'Vegetation') & (NDVI <= 0.4)")

            return {"Land_Cover": land_cover_rules, "Vegetation_Types": vegetation_rules}
    except Exception as e:
        st.error(f"Error creating example rule sets: {str(e)}")
        return {}


def render_segmentation_tab():
    """Render the segmentation tab for image segmentation and feature calculation."""
    st.header("Image Segmentation")
    st.write("Configure segmentation parameters and run the algorithm")

    col1, col2 = st.columns(2)

    with col1:
        scale_param = st.slider(
            "Scale Parameter",
            min_value=5,
            max_value=100,
            value=40,
            step=5,
            help="Controls the size of segments. Higher values create larger segments.",
        )
        compactness_param = st.slider(
            "Compactness",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Controls the compactness of segments. Higher values create more compact segments.",
        )

    with col2:
        segmentation_name = st.text_input("Segmentation Layer Name", "Base_Segmentation")

    st.subheader("Configure Band Mappings")
    st.write("Set up mappings for spectral bands to use in indices and analysis")

    # Get raw bands from the image data
    raw_bands = [f"band_{i + 1}" for i in range(st.session_state.image_data.shape[0])]

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.band_mappings["blue"] = st.selectbox(
            "Blue band mapping",
            raw_bands,
            index=raw_bands.index(st.session_state.band_mappings.get("blue", raw_bands[0]).replace("_mean", ""))
            if "blue" in st.session_state.band_mappings
            else 0,
        )
        st.session_state.band_mappings["green"] = st.selectbox(
            "Green band mapping",
            raw_bands,
            index=raw_bands.index(
                st.session_state.band_mappings.get("green", raw_bands[min(1, len(raw_bands) - 1)]).replace("_mean", "")
            )
            if "green" in st.session_state.band_mappings
            else min(1, len(raw_bands) - 1),
        )
    with col2:
        st.session_state.band_mappings["red"] = st.selectbox(
            "Red band mapping",
            raw_bands,
            index=raw_bands.index(
                st.session_state.band_mappings.get("red", raw_bands[min(2, len(raw_bands) - 1)]).replace("_mean", "")
            )
            if "red" in st.session_state.band_mappings
            else min(2, len(raw_bands) - 1),
        )
        if len(raw_bands) > 3:
            st.session_state.band_mappings["nir"] = st.selectbox(
                "NIR band mapping",
                raw_bands,
                index=raw_bands.index(
                    st.session_state.band_mappings.get("nir", raw_bands[min(3, len(raw_bands) - 1)]).replace("_mean", "")
                )
                if "nir" in st.session_state.band_mappings
                else min(3, len(raw_bands) - 1),
            )

    for key in st.session_state.band_mappings:
        if not st.session_state.band_mappings[key].endswith("_mean"):
            st.session_state.band_mappings[key] = f"{st.session_state.band_mappings[key]}_mean"

    segmentation_button = st.button("Run Segmentation")

    if segmentation_button:
        segmentation_layer = perform_segmentation(
            st.session_state.image_data,
            st.session_state.transform,
            st.session_state.crs,
            scale_param,
            compactness_param,
            segmentation_name,
        )

        if segmentation_layer:
            fig = plot_layer(
                segmentation_layer,
                st.session_state.image_data,
                rgb_bands=(3, 2, 1),  # Adjusted for 0-indexed bands
                show_boundaries=True,
            )
            st.pyplot(fig)

            st.success(f"Segmentation '{segmentation_name}' completed successfully!")

    st.subheader("Calculate Features")

    if not st.session_state.layers:
        st.warning("No segmentation layers available. Run segmentation first.")
    else:
        segmentation_for_features = st.selectbox(
            "Select segmentation layer for feature calculation:", options=list(st.session_state.layers.keys())
        )

        if segmentation_for_features:
            selected_layer = st.session_state.layers[segmentation_for_features]

            # Get attributes available from the selected layer
            layer_attributes = get_layer_attributes(segmentation_for_features)

            # Find band attributes in the layer
            band_attributes = [attr for attr in layer_attributes if attr.startswith("band_") and attr.endswith("_mean")]

            feature_options = st.multiselect(
                "Select features to calculate:", ["NDVI", "Spectral Indices", "Shape Metrics"], default=[]
            )

            if "NDVI" in feature_options:
                st.subheader("NDVI Configuration")
                col1, col2, col3 = st.columns(3)

                with col1:
                    nir_column = st.selectbox(
                        "NIR column:",
                        band_attributes,
                        index=band_attributes.index(st.session_state.band_mappings.get("nir", "band_4_mean"))
                        if "nir" in st.session_state.band_mappings and st.session_state.band_mappings["nir"] in band_attributes
                        else 0,
                    )
                with col2:
                    red_column = st.selectbox(
                        "RED column:",
                        band_attributes,
                        index=band_attributes.index(st.session_state.band_mappings.get("red", "band_3_mean"))
                        if "red" in st.session_state.band_mappings and st.session_state.band_mappings["red"] in band_attributes
                        else 0,
                    )
                with col3:
                    ndvi_output = st.text_input("Output column name:", "NDVI")

            if st.button("Calculate Selected Features"):
                features_calculated = False

                if "NDVI" in feature_options:
                    if calculate_ndvi(selected_layer, nir_column, red_column, ndvi_output):
                        fig = plot_layer(selected_layer, attribute=ndvi_output, title=f"{ndvi_output} Values", cmap="RdYlGn")
                        st.pyplot(fig)
                        st.success(f"{ndvi_output} calculation completed!")
                        features_calculated = True
                        layer_attributes = get_layer_attributes(segmentation_for_features)

                if "Spectral Indices" in feature_options:
                    indices_band_mappings = {
                        "blue": st.session_state.band_mappings.get("blue", "band_1_mean"),
                        "green": st.session_state.band_mappings.get("green", "band_2_mean"),
                        "red": st.session_state.band_mappings.get("red", "band_3_mean"),
                        "nir": st.session_state.band_mappings.get("nir", "band_4_mean")
                        if "nir" in st.session_state.band_mappings
                        else "band_4_mean",
                    }

                    if calculate_spectral_indices(selected_layer, indices_band_mappings):
                        st.success("Spectral indices calculation completed!")
                        features_calculated = True

                        layer_attributes = get_layer_attributes(segmentation_for_features)

                        indices = [
                            col
                            for col in layer_attributes
                            if col not in ["geometry", "segment_id", ndvi_output] and not col.startswith("band_")
                        ]
                        if indices:
                            st.write("Available spectral indices:")
                            st.write(", ".join(indices))

                            selected_index = st.selectbox("Visualize index:", indices)
                            if selected_index:
                                fig = plot_layer(selected_layer, attribute=selected_index, title=f"{selected_index} Values")
                                st.pyplot(fig)

                if "Shape Metrics" in feature_options:
                    if calculate_shape_metrics(selected_layer):
                        st.success("Shape metrics calculation completed!")
                        features_calculated = True

                        layer_attributes = get_layer_attributes(segmentation_for_features)

                        metrics = [
                            col
                            for col in layer_attributes
                            if col not in ["geometry", "segment_id"] and not col.startswith("band_") and col != ndvi_output
                        ]
                        if metrics:
                            st.write("Available shape metrics:")
                            st.write(", ".join(metrics))

                if features_calculated:
                    st.success("All selected features calculated successfully!")


def render_classification_tab():
    """Render the classification tab for applying rule sets to segmentation layers."""
    st.header("Classification")

    if not st.session_state.layers:
        st.warning("No segmentation layers available. Run segmentation first.")
    else:
        st.subheader("Apply Rule Sets")

        if not st.session_state.rule_sets:
            st.info("No rule sets defined yet. Go to Rule Builder tab to create rule sets.")
        else:
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                selected_ruleset = st.selectbox("Select rule set to apply:", options=list(st.session_state.rule_sets.keys()))

            with col2:
                input_layer = st.selectbox("Select input layer:", options=list(st.session_state.layers.keys()))

            with col3:
                result_field = st.text_input("Result field name:", "classification")

            if selected_ruleset and input_layer:
                output_layer_name = st.text_input("Output layer name:", f"{selected_ruleset}_results")

                if st.button("Apply Rule Set"):
                    ruleset = st.session_state.rule_sets[selected_ruleset]
                    input_layer_obj = st.session_state.layers[input_layer]
                    result_layer = apply_rule_set(ruleset, input_layer_obj, output_layer_name, result_field)

                    if result_layer:
                        fig = plot_classification(result_layer, class_field=result_field)
                        st.pyplot(fig)

                        area_stats = calculate_area_stats(result_layer, result_field)

                        if area_stats:
                            st.subheader("Area Statistics")
                            stats_data = []
                            for class_name, area in area_stats.get("class_areas", {}).items():
                                percentage = area_stats.get("class_percentages", {}).get(class_name, 0)
                                stats_data.append(
                                    {"Class": class_name, "Area (sq. units)": f"{area:.2f}", "Percentage": f"{percentage:.1f}%"}
                                )

                            st.table(stats_data)

                        st.success(f"Rule set '{selected_ruleset}' applied successfully to create layer '{output_layer_name}'!")

        st.subheader("Load Example Rule Sets")

        if st.button("Load Vegetation Classification Example"):
            has_ndvi = False
            for layer_name, layer in st.session_state.layers.items():
                if "NDVI" in layer.objects.columns:
                    has_ndvi = True
                    break

            if not has_ndvi:
                st.warning("NDVI not calculated for any layer. Please calculate NDVI first in the Segmentation tab.")
            else:
                example_rule_sets = create_example_rule_sets()

                if example_rule_sets:
                    st.session_state.rule_sets.update(example_rule_sets)
                    st.success("Example rule sets loaded successfully!")


def render_rule_builder_tab():
    """Render the rule builder tab for creating and managing classification rule sets."""
    st.header("Rule Builder")
    st.write("Create and manage classification rule sets and rules")

    st.subheader("Create Rule Set")

    new_ruleset_name = st.text_input("New Rule Set Name:", "")
    if new_ruleset_name and st.button("Create New Rule Set"):
        if new_ruleset_name in st.session_state.rule_sets:
            st.warning(f"Rule set '{new_ruleset_name}' already exists.")
        else:
            with st.spinner("Creating new rule set..."):
                st.session_state.rule_sets[new_ruleset_name] = RuleSet(name=new_ruleset_name)
                st.session_state.active_ruleset = new_ruleset_name
                st.success(f"Rule set '{new_ruleset_name}' created successfully!")

    st.subheader("Manage Rules")

    if not st.session_state.rule_sets:
        st.info("No rule sets created yet. Create a rule set first.")
    else:
        ruleset_selection = st.selectbox("Select rule set to manage:", options=list(st.session_state.rule_sets.keys()))

        if ruleset_selection:
            st.session_state.active_ruleset = ruleset_selection
            active_ruleset = st.session_state.rule_sets[ruleset_selection]

            st.write(f"Rules in '{ruleset_selection}':")

            rules = active_ruleset.get_rules()
            if not rules:
                st.info("No rules in this rule set yet.")
            else:
                rule_data = []
                for i, (name, condition) in enumerate(rules):
                    rule_data.append({"#": i + 1, "Name": name, "Condition": condition})

                st.table(pd.DataFrame(rule_data))

            st.subheader("Add New Rule")

            if not st.session_state.available_attributes:
                st.warning("No layers with attributes available. Create a segmentation layer with features first.")
            else:
                st.write("Available attributes for rules:")
                st.write(", ".join(sorted(st.session_state.available_attributes)))

                col1, col2 = st.columns(2)

                with col1:
                    rule_name = st.text_input("Rule Name:", "")

                st.subheader("Rule Condition Builder")

                if "condition_builder" not in st.session_state:
                    st.session_state.condition_builder = []

                if st.button("Add Condition Component"):
                    st.session_state.condition_builder.append({"attribute": "", "operator": ">", "value": "", "connector": "&"})

                condition_parts = []

                for i, condition in enumerate(st.session_state.condition_builder):
                    st.subheader(f"Condition Component {i + 1}")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        attribute = st.selectbox(
                            f"Attribute {i + 1}", options=sorted(st.session_state.available_attributes), key=f"attr_{i}"
                        )
                        st.session_state.condition_builder[i]["attribute"] = attribute

                    with col2:
                        operator = st.selectbox(f"Operator {i + 1}", options=[">", ">=", "<", "<=", "==", "!="], key=f"op_{i}")
                        st.session_state.condition_builder[i]["operator"] = operator

                    with col3:
                        if operator == "==" or operator == "!=":
                            value = st.text_input(f"Value {i + 1} (use quotes for text)", key=f"val_{i}")
                        else:
                            value = st.number_input(f"Value {i + 1}", key=f"val_{i}", format="%.2f")
                        st.session_state.condition_builder[i]["value"] = value

                    condition_part = f"{attribute} {operator} {value}"
                    condition_parts.append(condition_part)

                    if i < len(st.session_state.condition_builder) - 1:
                        connector = st.selectbox("Connect with", options=["&", "|"], key=f"conn_{i}")
                        st.session_state.condition_builder[i]["connector"] = connector

                if condition_parts:
                    final_condition = ""
                    for i, part in enumerate(condition_parts):
                        final_condition += part
                        if i < len(condition_parts) - 1:
                            final_condition += f" {st.session_state.condition_builder[i]['connector']} "

                    st.subheader("Final Condition:")
                    st.code(final_condition)

                    manual_condition = st.text_area("Or manually edit condition:", final_condition)

                    if manual_condition and st.button("Add Rule to Set"):
                        if not rule_name:
                            st.warning("Rule Name is required.")
                        else:
                            with st.spinner("Adding rule..."):
                                active_ruleset.add_rule(name=rule_name, condition=manual_condition)
                                st.session_state.condition_builder = []
                                st.success(f"Rule '{rule_name}' added to rule set '{ruleset_selection}'!")
                                st.rerun()

                if st.button("Clear Condition Builder"):
                    st.session_state.condition_builder = []
                    st.rerun()

            st.subheader("Delete Rule Set")
            if st.button("Delete Current Rule Set", key="delete_ruleset"):
                with st.spinner("Deleting rule set..."):
                    if st.session_state.active_ruleset in st.session_state.rule_sets:
                        del st.session_state.rule_sets[st.session_state.active_ruleset]
                        if st.session_state.rule_sets:
                            st.session_state.active_ruleset = list(st.session_state.rule_sets.keys())[0]
                        else:
                            st.session_state.active_ruleset = None
                        st.success(f"Rule set '{ruleset_selection}' deleted.")
                        st.rerun()


def render_layer_manager_tab():
    """Render the layer manager tab for managing segmentation and classification layers."""
    st.header("Layer Manager")
    st.write("View, inspect, and manage layers")

    st.subheader("Available Layers")

    if not st.session_state.layers:
        st.info("No layers created yet.")
    else:
        layer_list = st.session_state.manager.get_layer_names()

        for i, layer_name in enumerate(layer_list):
            st.markdown(f"**{i + 1}. {layer_name}**")

            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                if st.button("View Info", key=f"info_{layer_name}"):
                    layer = st.session_state.layers.get(layer_name)
                    with st.spinner("Loading layer information..."):
                        if layer and hasattr(layer, "objects"):
                            st.write(f"Layer: {layer_name}")
                            st.write(f"Number of features: {len(layer.objects)}")
                            st.write("Attributes:")
                            attribute_list = [col for col in layer.objects.columns if col != "geometry"]
                            st.write(", ".join(attribute_list))

            with col2:
                if st.button("Visualize", key=f"vis_{layer_name}"):
                    layer = st.session_state.layers.get(layer_name)
                    with st.spinner("Generating visualization..."):
                        if layer:
                            for field in ["classification", "veg_class"]:
                                if field in layer.objects.columns:
                                    fig = plot_classification(layer, class_field=field)
                                    st.pyplot(fig)
                                    break
                            else:
                                fig = plot_layer(
                                    layer,
                                    st.session_state.image_data,
                                    rgb_bands=(2, 1, 0),  # Adjusted for 0-indexed bands
                                    show_boundaries=True,
                                )
                                st.pyplot(fig)

            with col3:
                if st.button("Export Options", key=f"export_{layer_name}"):
                    layer = st.session_state.layers.get(layer_name)
                    if layer:
                        st.session_state.export_layer = layer_name
                        st.info(f"Layer '{layer_name}' added to export queue. Go to Results tab to export.")

            with col4:
                if st.button("❌", key=f"del_{layer_name}"):
                    if layer_name in st.session_state.layers:
                        with st.spinner(f"Deleting layer '{layer_name}'..."):
                            del st.session_state.layers[layer_name]
                            if layer_name in layer_list:
                                st.session_state.manager.remove_layer(layer_name)
                            st.success(f"Layer '{layer_name}' deleted!")
                            update_available_attributes()
                            st.rerun()

            st.markdown("---")


def render_results_tab():
    """Render the results tab for exporting segmentation and classification results."""
    st.header("Results and Export")

    if not st.session_state.layers:
        st.warning("No results to display yet. Please run segmentation and classification first.")
    else:
        st.write("Export your results as vector (GeoJSON) or raster (GeoTIFF) files.")

        st.subheader("Available Layers")
        for i, layer_name in enumerate(st.session_state.manager.get_layer_names()):
            st.write(f"  {i + 1}. {layer_name}")

        st.subheader("Export Options")

        export_layer_name = st.selectbox(
            "Select Layer to Export",
            options=list(st.session_state.layers.keys()),
            format_func=lambda x: x.capitalize().replace("_", " "),
        )

        if export_layer_name:
            export_path = st.text_input("Export Filename Base (without extension):", export_layer_name.lower())

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export as Vector (GeoJSON)"):
                    export_filepath = os.path.join(st.session_state.output_dir, f"{export_path}.geojson")
                    success = export_vector(st.session_state.layers[export_layer_name], export_filepath)

                    if success:
                        with open(export_filepath, "rb") as f:
                            st.download_button(
                                label="Download GeoJSON", data=f, file_name=f"{export_path}.geojson", mime="application/geo+json"
                            )

            with col2:
                column_options = get_layer_attributes(export_layer_name)

                if column_options:
                    raster_column = st.selectbox("Select attribute for raster export:", column_options)

                    if st.button("Export as Raster (GeoTIFF)"):
                        export_filepath = os.path.join(st.session_state.output_dir, f"{export_path}.tif")
                        success = export_raster(st.session_state.layers[export_layer_name], export_filepath, raster_column)

                        if success:
                            with open(export_filepath, "rb") as f:
                                st.download_button(
                                    label="Download GeoTIFF", data=f, file_name=f"{export_path}.tif", mime="image/tiff"
                                )

            st.subheader("Project Export")

            if st.button("Export Project Metadata"):
                with st.spinner("Creating project metadata..."):
                    metadata = {
                        "project_name": "nickyspatial Project",
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "layers": list(st.session_state.manager.get_layer_names()),
                        "rule_sets": {},
                    }

                    for ruleset_name, ruleset in st.session_state.rule_sets.items():
                        metadata["rule_sets"][ruleset_name] = [
                            {"name": name, "condition": condition} for name, condition in ruleset.get_rules()
                        ]

                    import json

                    metadata_filepath = os.path.join(st.session_state.output_dir, "project_metadata.json")
                    with open(metadata_filepath, "w") as f:
                        json.dump(metadata, f, indent=4)

                    with open(metadata_filepath, "rb") as f:
                        st.download_button(
                            label="Download Project Metadata", data=f, file_name="project_metadata.json", mime="application/json"
                        )


def render_welcome_screen():
    """Render the welcome screen for the application."""
    st.write("""
    ## Welcome to the nickyspatial Interactive Web App

    This application allows you to perform object-based image analysis on remote sensing data with complete flexibility
    to configure all parameters, rules, and layers.

    To get started:
    1. Use the sidebar to upload your own raster data or use the sample data
    2. Configure segmentation parameters and create segments
    3. Calculate features like NDVI, spectral indices, or shape metrics
    4. Create rule sets and classification rules in the Rule Builder
    5. Apply your rules to classify the image
    6. Visualize and export your results

    ### Features:
    - Fully configurable image segmentation parameters
    - Dynamic creation of custom rule sets and rules
    - Flexible band mapping for spectral indices
    - Multiple layer management and visualization
    - Export results in vector and raster formats
    """)

    st.image(
        "https://cdn.pixabay.com/photo/2018/04/11/19/52/satellite-3311245_960_720.jpg", caption="Remote Sensing Image Analysis"
    )


def render_app_info():
    """Render the application information in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About nickyspatial**

    An open-source object-based image analysis library for remote sensing.

    [GitHub Repository](https://github.com/kshitijrajsharma/nickyspatial)
    """)


def main():
    """Main function to run the Streamlit app."""
    initialize_session_state()

    st.title("nickyspatial - Object-Based Image Analysis")
    st.markdown("An interactive tool for remote sensing analysis and classification")

    st.sidebar.header("Configuration")

    st.sidebar.subheader("1. Load Data")
    data_option = st.sidebar.radio("Choose data source:", ("Upload your own", "Use sample data"))

    if data_option == "Upload your own":
        uploaded_file = st.sidebar.file_uploader("Upload a raster file (GeoTIFF format)", type=["tif", "tiff"])
        if uploaded_file is not None:
            temp_file = os.path.join(st.session_state.output_dir, "uploaded_file.tif")
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            load_success = load_raster(temp_file)
    else:
        sample_data_path = st.sidebar.text_input("Path to sample data:", "data/sample.tif")
        if st.sidebar.button("Load Sample Data"):
            load_success = load_raster(sample_data_path)

    if st.session_state.image_data is not None:
        st.sidebar.success("Image loaded successfully")
        st.sidebar.info(f"Image dimensions: {st.session_state.image_data.shape}")
        if st.session_state.crs:
            st.sidebar.info(f"CRS: {st.session_state.crs}")

        tabs = st.tabs(["Segmentation", "Classification", "Rule Builder", "Layer Manager", "Results"])

        with tabs[0]:
            render_segmentation_tab()

        with tabs[1]:
            render_classification_tab()

        with tabs[2]:
            render_rule_builder_tab()

        with tabs[3]:
            render_layer_manager_tab()

        with tabs[4]:
            render_results_tab()
    else:
        render_welcome_screen()

    render_app_info()


if __name__ == "__main__":
    main()
