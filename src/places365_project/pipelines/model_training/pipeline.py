"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import initialize, train_model



def create_pipeline():
    return pipeline([
        node(
            initialize,
            ["params:processed_path", "params:batch_size",'params:lr', 'params:patience', 'params:frequency', 'params:no_classes'],
            ['model', 'sets'],
            name="initialize_model"
        ),
        node(
            train_model,
            ["model", "sets", "params:models_path"],
            None,
            name="train_model"
        )
    ])
