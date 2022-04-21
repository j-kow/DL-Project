"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_model_and_data, test_model


def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            load_model_and_data,
            ["params:models_path", "params:final_path", 'params:batch_size', 'params:dataloader_num_workers', "params:model_checkpoint"],
            ['test_model', 'test_set'],
            name="load_model_and_data"
        ),
        node(
            test_model,
            ["test_model", "test_set", "params:image_path", "params:figsize"],
            ["accuracy", "cf_matrix"],
            name="test_model"
        )
    ])
