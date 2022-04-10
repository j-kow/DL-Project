"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from places365_project.pipelines import dataset_creation as dc
from places365_project.pipelines import model_training as mt


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    dataset_creation_pipeline = dc.create_pipeline()
    model_training_pipeline = mt.create_train_pipeline()
    gridsearch_pipeline = mt.create_grisdearch_pipeline()

    return {
        "dc": dataset_creation_pipeline,
        "mt": model_training_pipeline,
        "gs": gridsearch_pipeline,
        "__default__": dataset_creation_pipeline
    }
