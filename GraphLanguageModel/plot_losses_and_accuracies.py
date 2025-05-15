import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="plot_losses_and_accuracies", usage="%(prog)s [options]")
parser.add_argument(
   "losses_path",
    help="The path to the losses.csv file of a trained model.",
    type=str,
)
args = parser.parse_args()

def main(parsed_args):
   losses_path = Path(parsed_args.losses_path).resolve()
   losses = pd.read_csv(losses_path, sep=";", usecols=["loss", "last_dev_acc"])
   losses.last_dev_acc *= 100
   losses.rename(columns={"last_dev_acc": "Test Accuracy", "loss": "Loss"}, inplace=True)
   axes = losses.plot(xlabel="Batch", secondary_y="Loss")
   axes.set_ylabel("Accuracy in %")
   plt.show()
   

if __name__ == "__main__":
   main(args) 