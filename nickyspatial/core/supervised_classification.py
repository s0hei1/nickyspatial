# -*- coding: utf-8 -*-
import warnings

import geopandas as gpd
import numpy as np
import rasterio.features
from shapely.geometry import Polygon
from skimage import segmentation

from .layer import Layer
from sklearn.ensemble import RandomForestClassifier


class SupervisedClassification:
    #TODO: name vs layer_name
    """Implementation of Supervised Classification algorithm.
    """

    def __init__(self, name=None, classifier_type="Random Forest", classifier_params={}):
        """Initialize the segmentation algorithm.

        Parameters:
        -----------
        scale : str
            classifier type name eg: RF for Random Forest, SVC for Support Vector Classifier
        classifier_params : dict
           additional parameters relayed to classifier
        """
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params
        self.training_layer=None
        self.classifier=None
        self.name = name if name else "Supervised_Classification"
        # self.samples=None

    def _training_sample(self, layer, samples):
        """Create vector objects from segments.

        Parameters:
        -----------
        samples : dict
            key: class_name
            values: list of segment_ids
            eg: {"cropland":[1,2,3],"built-up":[4,5,6]}

        Returns:
        --------
        segment_objects : geopandas.GeoDataFrame
            GeoDataFrame with segment polygons
        """

        # cropland_idxs=[11,396,489,1400,3818,1532,1470,810,594,1224,1388,13] #class=1
        # meadow_idxs=[376,1040,44,2229,1099,514,65,3533,983,807] #class=2
        # training_layer=source_layer.objects
        # training_layer=layer 
        layer['classification'] = ''

        for class_name in samples.keys():
            layer.loc[layer['segment_id'].isin(samples[class_name]), 'classification'] = class_name

        layer = layer[layer['classification'] != '']
        self.training_layer=layer
        return layer


    def _train(self):
        """Calculate statistics for segments based on image data.

        Parameters:
        -----------
        layer : Layer
            Layer containing segments
        image_data : numpy.ndarray
            Array with raster data values (bands, height, width)
        bands : list of str
            Names of the bands
        """
        X = self.training_layer.drop(columns=['segment_id', 'classification','geometry'],errors="ignore")
        y = self.training_layer['classification']

        if self.classifier_type == "Random Forest":
            self.classifier = RandomForestClassifier(**self.classifier_params)
            
            self.classifier.fit(X, y)
        
        print("OOB Score:", self.classifier.oob_score_)

        return self.classifier


    def _prediction(self, layer):
        layer["classification"]=""
        print(layer.columns)
        X = layer.drop(columns=['segment_id', 'classification','geometry'],errors="ignore")

        predictions = self.classifier.predict(X)
        layer.loc[layer['classification'] == "", 'classification'] = predictions
        return layer

    def execute(self, source_layer,samples={},layer_manager=None,layer_name=None,):
       
        result_layer = Layer(name=layer_name, parent=source_layer, type="merged")
        result_layer.transform = source_layer.transform
        result_layer.crs = source_layer.crs
        result_layer.raster = source_layer.raster.copy() if source_layer.raster is not None else None
        
        layer=source_layer.objects.copy()
        self._training_sample(layer, samples)
        self._train()
        layer=self._prediction(layer)

        result_layer.objects = layer

        result_layer.metadata = {
            "supervised classification": self.name,
        }

        if layer_manager:
            layer_manager.add_layer(result_layer)

        return result_layer
    

