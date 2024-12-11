import numpy as np
from resspect.feature_extractors.light_curve import LightCurve


class ExampleFeatureExtractor(LightCurve):

    # The list of feature names that will be extracted from the light curve.
    feature_names = [
        'anderson_darling_normal',
        'chi2',
        'stetson_K',
        'weighted_mean',
        # ... whatever additional features are extracted
    ]

    # The id column of the data. e.g. 'ID', 'obj_id', etc.
    id_column = "id" 

    # The name of the label column. e.g. 'type', 'label', 'sntype', etc.
    label_column = "type"

    # The classes that are not considered anomalies. e.g. ['Ia', 'Nomral', etc.]
    non_anomaly_classes = ["Ia"]


    def __init__(self):
        super().__init__()
        self.num_features = len(ExampleFeatureExtractor.feature_names)
        self.features = None


    @classmethod
    def get_features(cls, filters: list) -> list[str]:
        """
        A class method that returns the list of features that will be extracted
        from the light curve.

        Often the feature list is a cross product of the feature name and the filters.
        In this example we return only the feature name list.

        Returns
        -------
        feature_names: list[str]
            List of feature names that will be extracted from the light curve.
        """
        return cls.feature_names


    @classmethod
    def get_metadata_columns(cls, **kwargs) -> list[str]:
        """
        A class method that returns the metadata columns for the feature extractor.
        Depending on how dynamic the metadata columns are, this method can be
        a hard-coded list or a dynamically generated list.

        Returns
        -------
        metadata_columns: list[str]
            List of metadata columns for the feature extractor.
        """

        # hard-coded example
        metadata_columns = [cls.id_column, "redshift", cls.label_column, "sncode", "sample"]

        # dynamic example
        kwargs['override_primary_columns'] = [cls.id_column, "redshift", cls.label_column, "sncode", "sample"]
        metadata_columns = super().get_metadata_header(**kwargs)

        return metadata_columns


    @classmethod
    def get_feature_header(cls, filters: list[str], **kwargs) -> list[str]:
        """
        A class method that returns the full list column names for an output file.
        This includes the metadata columns and the feature columns.

        Returns
        -------
        header: list[str]
            List of column names for the output file.
        """

        # One way to return this is to concatenate the metadata columns and the feature columns
        return ExampleFeatureExtractor.get_metadata_header(**kwargs) + ExampleFeatureExtractor.get_features(filters)


    def fit_all(self) -> np.ndarray:
        """
        Extracts features for all light curves in the dataset.

        Returns
        -------
        features: np.ndarray
            Features extracted from the light curves.
        """
        # Implement feature extraction here
        self.features = self.example_extraction_function()
        return self.features


    def _example_extraction_function():
        # Just for demo purposes
        pass


    def get_features_to_write(self):
        # Implement this method to return the features that will be written to 
        # the output file.
        # Base class has a simple implementation, but you can override it here.

        # The base class implementation will return `features_list`:
        #  features_list = [
        #     self.id,
        #     self.redshift,
        #     self.sntype,
        #     self.sncode,
        #     self.sample]
        # features_list.extend(self.features)
        return super().get_features_to_write()
