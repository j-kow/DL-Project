from kedro.pipeline import node, pipeline

from .nodes import prune, refactor_data_structure, split_into_train_val_test


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
            split_into_train_val_test,
            ["params:processed_path", "params:final_path", "params:split_required", "params:delete_unsplitted_data"],
            None,
            name="split_dataset"
        ),
#        node(
#            create_data_loader,
#            ["params:processed_path", "params:train_val_test", "params:batch_size", "params:shuffle", "params:dataloader_num_workers"],
#            "places_dataset",
#            name="create_data_loader"
#        )
    ])
