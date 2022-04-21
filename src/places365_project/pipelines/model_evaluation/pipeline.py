"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import initialize, train_model, create_gridsearch_parameters, run_gridsearch


def create_train_pipeline() -> Pipeline:
    return pipeline([
        node(
            initialize,
            ["params:final_path", "params:batch_size", 'params:lr', 'params:patience', 'params:frequency',
             'params:workers', 'params:no_classes'],
            ['model', 'sets'],
            name="initialize_model"
        ),
        node(
            train_model,
            ["model", "sets", "params:max_epochs", "params:models_path", "params:frequency", "params:earlystop_patience"],
            "trainer",
            name="train_model"
        )
    ])


def create_grisdearch_pipeline() -> Pipeline:
    return pipeline([
        node(
            create_gridsearch_parameters,
            None,
            "grid_params",
            name="create_gridsearch_parameters"
        ),
        node(
            run_gridsearch,
            ["params:final_path", "params:batch_size", 'params:lr', 'params:patience', 'params:frequency',
             'params:workers', 'params:no_classes', "params:max_epochs", "params:models_path", "grid_params"],
            None,
            name="run_gridsearch"
        )
    ])
