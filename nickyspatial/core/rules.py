"""Provides a rule engine for object-based analysis, where segments or layers are processed according to custom logic.

Main idea here is to allow encode expert rules that can be applied to object segments which are layers in a nickyspatial context.
So rules are tied up to the layers , they can be attached or revoked or executed multiple items

Developers can define domain-specific rules to classify or merge features based on attributes.
This module includes the Rule and RuleSet classes, which allow users to create, manage, and apply rules to layers.
The RuleSet class can be used to group multiple rules together, and the execute method applies these rules to a given layer.
The rules can be defined using string expressions that can be evaluated using the numexpr library for performance.
"""

import numexpr as ne
import pandas as pd

from .layer import Layer


class Rule:
    """A rule defines a condition to classify segments."""

    def __init__(self, name, condition, class_value=None):
        """Initialize a rule.

        Parameters:
        -----------
        name : str
            Name of the rule
        condition : str
            Condition as a string expression that can be evaluated using numexpr
        class_value : str, optional
            Value to assign when the condition is met.
            If None, uses the rule name.
        """
        self.name = name
        self.condition = condition
        self.class_value = class_value if class_value is not None else name

    def __str__(self):
        """String representation of the rule."""
        return f"Rule '{self.name}': {self.condition} -> {self.class_value}"


class RuleSet:
    """A collection of rules to apply to a layer."""

    def __init__(self, name=None):
        """Initialize a rule set.

        Parameters:
        -----------
        name : str, optional
            Name of the rule set
        """
        self.name = name if name else "RuleSet"
        self.rules = []

    @staticmethod
    def wrap_condition_parts_simple(self, condition):
        """Wrap condition parts with parentheses for evaluation."""
        parts = condition.split("&")
        parts = [f"({part.strip()})" for part in parts]
        return " & ".join(parts)

    def add_rule(self, name, condition, class_value=None):
        """Add a rule to the rule set.

        Parameters:
        -----------
        name : str
            Name of the rule
        condition : str
            Condition as a string expression that can be evaluated using numexpr
        class_value : str, optional
            Value to assign when the condition is met

        Returns:
        --------
        rule : Rule
            The added rule
        """
        rule = Rule(name, condition, class_value)
        self.rules.append(rule)
        return rule

    def execute(
        self,
        source_layer,
        layer_manager=None,
        layer_name=None,
        result_field="classification",
    ):
        """Apply rules to classify segments in a layer.

        Parameters:
        -----------
        source_layer : Layer
            Source layer with segments to classify
        layer_manager : LayerManager, optional
            Layer manager to add the result layer to
        layer_name : str, optional
            Name for the result layer
        result_field : str
            Field name to store classification results

        Returns:
        --------
        result_layer : Layer
            Layer with classification results
        """
        if not layer_name:
            layer_name = f"{source_layer.name}_{self.name}"

        result_layer = Layer(name=layer_name, parent=source_layer, type="classification")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None

        result_layer.objects = source_layer.objects.copy()

        if result_field not in result_layer.objects.columns:
            result_layer.objects[result_field] = None

        result_layer.metadata = {
            "ruleset_name": self.name,
            "rules": [
                {
                    "name": rule.name,
                    "condition": rule.condition,
                    "class_value": rule.class_value,
                }
                for rule in self.rules
            ],
            "result_field": result_field,
        }

        for rule in self.rules:
            try:
                if result_field in result_layer.objects.columns and (
                    f"{result_field} ==" in rule.condition
                    or f"{result_field}==" in rule.condition
                    or f"{result_field} !=" in rule.condition
                    or f"{result_field}!=" in rule.condition
                ):
                    ## TODO : better way to handle this , because & searching in string is not a good idea,
                    # this might produce bug for complex rules
                    eval_condition = rule.condition.replace("&", " and ").replace("|", " or ")

                    mask = result_layer.objects.apply(
                        lambda row, cond=eval_condition: eval(
                            cond,
                            {"__builtins__": {}},
                            {col: row[col] for col in result_layer.objects.columns if col != "geometry"},
                        ),
                        axis=1,
                    )

                else:
                    try:
                        local_dict = {
                            col: result_layer.objects[col].values for col in result_layer.objects.columns if col != "geometry"
                        }

                        mask = ne.evaluate(rule.condition, local_dict=local_dict)
                        mask = pd.Series(mask, index=result_layer.objects.index).fillna(False)
                    except Exception:
                        mask = result_layer.objects.eval(rule.condition, engine="python")

                result_layer.objects.loc[mask, result_field] = rule.class_value

            except Exception as e:
                print(f"Error applying rule '{rule.name}': {str(e)}")
                continue

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer
