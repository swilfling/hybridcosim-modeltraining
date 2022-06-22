from typing import List
from ..interfaces import PickleInterface
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import TransformerMixin


class TransformerSet(TransformerMixin, PickleInterface):
    """
    This class implements a pipeline for feature transformers.
    """
    transformer_pipeline: Pipeline = None

    def __init__(self, transformers: List[TransformerMixin] = [], feature_names: List[str] = []):
        self.transformer_pipeline = make_pipeline(*transformers, 'passthrough')
        if len(self.transformer_pipeline.steps) > 1 and len(feature_names) > 0:
            self.transformer_pipeline.steps[0][1].feature_names_in_ = feature_names

    #################################### Getters and Setters ###########################################################

    def get_list_transfomers(self):
        return [transformer for (_, transformer) in self.transformer_pipeline.steps[:-1]]

    def get_transformers_of_type(self, type):
        return [transformer for (_, transformer) in self.transformer_pipeline.steps[:-1] if isinstance(transformer, type)]

    def get_transformer_by_index(self, index=0):
        # Omit last step of pipeline (is passthrough)
        if len(self.transformer_pipeline.steps) > 1:
            pipeline_steps = self.transformer_pipeline.steps[:-1]
            return pipeline_steps[index][1] if index < len(pipeline_steps) else None
        return None

    def get_transformer_by_name(self, name=""):
        return self.transformer_pipeline.named_steps.get(name, None)

    def set_feature_names(self, feature_names: List[str] = None):
        if len(self.transformer_pipeline.steps) > 1:
            self.transformer_pipeline.steps[0][1].feature_names_in_ = feature_names

    def get_feature_names_out(self, feature_names=None):
        return self.transformer_pipeline.get_feature_names_out(feature_names)

    def type_transf_full(self):
        return "_".join(transformer.__class__.__name__ for _, transformer in self.transformer_pipeline.steps[:-1])

    def type_last_transf(self, parent_type=None):
        if parent_type is None:
            if len(self.transformer_pipeline.steps) > 1:
                return self.transformer_pipeline.steps[-2][1].__class__.__name__
            else:
                return 'passthrough'
        else:
            return self.get_transformers_of_type(parent_type)[-1].__class__.__name__

    ########################################### Pipeline methods #######################################################

    def fit(self, X, y, **fit_params):
        self.transformer_pipeline.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        return self.transformer_pipeline.transform(X)
