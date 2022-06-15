from tensorflow import keras

from ModelTraining.datamodels.datamodels import Model
from ModelTraining.datamodels.datamodels.processing.datascaler import DataScaler
from ModelTraining.Utilities.Parameters import TrainingParams
from ModelTraining.datamodels.datamodels.wrappers.feature_extension import ExpandedModel, TransformerSet, FeatureExpansion


def train_model(model, x_train, y_train):
    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='mse',
              optimizer=optimizer)
    return model.fit(
        x_train, y_train,
        epochs=40,
        batch_size=24,
        validation_split=0.2,

    )


def create_model(training_params: TrainingParams, **kwargs):
    model = Model.from_name(training_params.model_type,
                            x_scaler_class=DataScaler.cls_from_name(training_params.normalizer),
                            name=training_params.str_target_feats(),
                            train_function=train_model, parameters={})
    expanders = FeatureExpansion.from_names(training_params.expansion, **kwargs.get('expander_parameters', {}))
    expanded_model = ExpandedModel(transformers=expanders, model=model)
    return expanded_model