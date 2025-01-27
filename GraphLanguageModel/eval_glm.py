import argparse

from utils.argparse import PathAction, problem_type_to_classification_bool
from GraphLanguageModel.pipelines import EvalPipeline
from GraphLanguageModel.pipelines.recipies import ModelRecipe


parser = argparse.ArgumentParser(prog="eval_glm", usage="%(prog)s [options]")
parser.add_argument("encoder_modelcard", 
                    help="The (huggingface modelhub) location of the encoder model.", 
                    type=str)
parser.add_argument("generator_modelcard", 
                    help="The (huggingface modelhub) location of the generator model.", 
                    type=str)
parser.add_argument("eval_file", 
                    help="Location of the preprocessed evaluation file.", 
                    type=str, action=PathAction)
parser.add_argument("-pt", "--problem_type", choices=["classification", "generation"], default="classification", type=str)
parser.add_argument("-gt", "--glm_type", help="Select whether to use a global or local GLM.", choices=["local", "global"], default="global", type=str)
parser.add_argument("-b", "--batch_size", default=64, type=int)
parser.add_argument("-r", "--repetitions", 
                    help="Determines how often the evaluation should be repeated.",
                    default=1, type=int)
parser.add_argument("-d", "--device", default='cpu', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    eval_data = args.eval_file
    model_recipe = ModelRecipe(args.encoder_modelcard, args.glm_type, args.generator_modelcard, gradient_checkpointing=False)
    eval_pipeline_builder = EvalPipeline.Builder().is_classification_task(problem_type_to_classification_bool(args.problem_type)).set_eval_data(eval_data).add_model_recipe(model_recipe).set_batch_size(args.batch_size).set_repetitions(args.repetitions).set_device(args.device)
    eval_pipeline = eval_pipeline_builder.build()
    del eval_pipeline_builder
    mean_score, score_std = eval_pipeline.eval()
    print(f"The repoted accuracy on {eval_data.name} is: {mean_score} with a standard deviation of {score_std}.")