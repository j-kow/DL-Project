from kedro.pipeline import node, pipeline

from .nodes import prune, refactor_data_structure, create_data_loader


def create_pipeline():
    return pipeline([
        node(
            refactor_data_structure,
            ["params:raw_path", "params:processed_path", "params:delete_raw_data"],
            None,
            name="refactor_data_structure"
        ),
        node(
            prune,
            ["params:processed_path", "params:prune_frac"],
            None,
            name="prune_dataset"
        ),
        node(
            create_data_loader,
            ["params:processed_path", "params:batch_size", "params:dataloader_num_workers"],
            "places_dataset",
            name="create_data_loader"
        )
    ])
