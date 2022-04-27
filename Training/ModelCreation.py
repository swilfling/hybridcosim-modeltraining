from tensorflow import keras

import ModelTraining.datamodels
from ModelTraining.Utilities.Parameters import TrainingParams


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


def create_expanders(expansion, **kwargs):
    expanders = []
    for expander_name in expansion:
        type = getattr(ModelTraining.datamodels.datamodels.processing, expander_name)
        expander = type(**kwargs.get('expander_parameters',None)) if expander_name == 'PolynomialExpansion' else type()
        expanders.append(expander)
    return expanders


def create_model(training_params: TrainingParams, **kwargs):
    model_type = getattr(ModelTraining.datamodels.datamodels, training_params.model_type)
    normalizer = getattr(ModelTraining.datamodels.datamodels.processing, training_params.normalizer)
    expanders = create_expanders(training_params.expansion, expander_parameters=kwargs.get('expander_parameters',{}))
    model = model_type(x_scaler_class=normalizer,
                      name="_".join(training_params.target_features),
                      train_function=train_model,
                      expanders=expanders,
                      parameters={})
    model.set_feature_names(kwargs.get('feature_names',None))
    return model