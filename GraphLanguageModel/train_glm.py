import argparse

from utils.argparse import KeywordAction, PathAction, problem_type_to_classification_bool
from GraphLanguageModel.pipelines import TrainPipeline
from GraphLanguageModel.pipelines.recipies import ModelRecipe, TrainRecipe


parser = argparse.ArgumentParser(prog="train_glm", usage="%(prog)s [options]")
parser.add_argument("encoder_modelcard", 
                    help="The (huggingface modelhub) location of the encoder model.", 
                    type=str)
parser.add_argument("generator_modelcard", 
                    help="The (huggingface modelhub) location of the generator model.", 
                    type=str)
parser.add_argument("train_file", 
                    help="Location of the preprocessed train file.", 
                    type=str, action=PathAction)
parser.add_argument("save_location", 
                    help="Location of the saved model and training information.", 
                    type=str, action=PathAction)
parser.add_argument("-pt", "--problem_type", choices=["classification", "generation"], default="classification", type=str)
parser.add_argument("-gt", "--glm_type", help="Select whether to use a global or local GLM.", choices=["local", "global"], default="global", type=str)
parser.add_argument("-d", "--device", default="cpu", type=str)
parser.add_argument("-b", "--batch_size", default=64, type=int)
optimizer_group = parser.add_argument_group("Optimizer")
optimizer_group.add_argument("-o", "--optimizer", 
                                choices=["Adadelta", "Adafactor", "Adagrad", "Adam", "AdamW", "SparseAdam",
                                        "Adamax", "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop",
                                        "Rprop", "SGD"],
                                default="AdamW",
                                type=str,
                                help="The Pytorch optimizer class used for training"
                            )
optimizer_group.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
optimizer_group.add_argument("optimizer_kwargs", 
                             help="Keyword arguments of the selected optimizer.", 
                             nargs="*", action=KeywordAction)
parser.add_argument("-ne", "--num_epochs", default=5, type=int)
parser.add_argument("-es", "--early_stopping", 
                    help="How many attempts at unsuccessful training epochs before early stopping.", 
                    default=2, type=int)
parser.add_argument("-ns", "--neighborhood_size", 
                    help="The maximum size of a given neighborhood in the knowledge graph around the input entity.", 
                    default=10, type=int)
parser.add_argument("-ef", "--eval_file", 
                    help="The location of the evaluation split file.", 
                    type=str, action=PathAction)
parser.add_argument("-c", "--checkpointing_interval", 
                    help="Determines after how many batches the script should save a model checkpoint, if appropriate.",
                    default=2000, type=int)
args = parser.parse_args()


if __name__ == "__main__":
    train_recipe = TrainRecipe(problem_type_to_classification_bool(args.problem_type), args.train_file, args.num_epochs, 
                               args.batch_size, args.early_stopping, args.optimizer, args.optimizer_kwargs, 
                               args.learning_rate, args.neighborhood_size)
    model_recipe = ModelRecipe(args.encoder_modelcard, args.glm_type, args.generator_modelcard)
    train_pipeline_builder = TrainPipeline.Builder().add_model_recipe(model_recipe).add_train_recipe(train_recipe).add_save_location(args.save_location).set_checkpointing_interval(args.checkpointing_interval)
    if args.eval_file is not None:
        train_pipeline_builder = train_pipeline_builder.set_eval_data(args.eval_file)
    train_pipeline = train_pipeline_builder.build()
    del train_pipeline_builder
    train_pipeline.train()
