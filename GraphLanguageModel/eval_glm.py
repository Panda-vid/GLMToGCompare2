from pathlib import Path

from GraphLanguageModel.pipelines import EvalPipeline
from GraphLanguageModel.pipelines.recipies import ModelRecipe


if __name__ == "__main__":
    eval_data = Path("./data/preprocessed/trex-dev-kilt.jsonl").resolve()
    encoder_modelcard = "./saved_models/trex/flan-t5-base/encoder"
    modelcard_generation = "./saved_models/trex/flan-t5-base/generator"
    model_recipe = ModelRecipe(encoder_modelcard, "global", modelcard_generation)
    eval_pipeline_builder = EvalPipeline.Builder().is_classification_task(True).set_eval_data(eval_data).add_model_recipe(model_recipe).set_device("cuda").set_batch_size(128).set_repetitions(5)
    eval_pipeline = eval_pipeline_builder.build()
    del eval_pipeline_builder
    mean_score, score_std = eval_pipeline.eval()
    print(f"The repoted accuracy on {eval_data.name} is: {mean_score} with a standard deviation of {score_std}.")