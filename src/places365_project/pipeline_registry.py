"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from places365_project.pipelines import dataset_creation as dc


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    dataset_creation_pipeline = dc.create_pipeline()

    return {
        "dc": dataset_creation_pipeline,
        "__default__": dataset_creation_pipeline
    }
