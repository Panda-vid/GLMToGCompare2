from pathlib import Path

from GraphLanguageModel.pipelines import TrainPipeline
from GraphLanguageModel.pipelines.recipies import ModelRecipe, TrainRecipe


if __name__ == "__main__":
    train_data = Path("./data/preprocessed/trex-train-kilt.jsonl").resolve()
    eval_data = Path("./data/preprocessed/trex-dev-kilt.jsonl").resolve()
    device = "cuda"
    batch_size = 42
    learning_rate = 1e-4
    optimizer = "AdamW"
    loss = "CrossEntropyLoss"
    num_epochs = 5
    early_stopping = 2
    classification = True
    encoder_modelcard = "./saved_models/trex/flan-t5-small/encoder"
    modelcard_generation = "./saved_models/trex/flan-t5-small/generator"
    neighborhood_size = 10
    save_location = Path("./saved_models/trex/flan-t5-small").resolve()

    train_recipe = TrainRecipe(True, train_data, num_epochs, batch_size, early_stopping, optimizer, learning_rate, loss, neighborhood_size)
    model_recipe = ModelRecipe(encoder_modelcard, "global", modelcard_generation)
    train_pipeline_builder = TrainPipeline.Builder().add_model_recipe(model_recipe).add_train_recipe(train_recipe).add_save_location(save_location).set_device("cuda").set_checkpointing_interval(2000).set_eval_data(eval_data)
    train_pipeline = train_pipeline_builder.build()
    del train_pipeline_builder
    train_pipeline.train()
