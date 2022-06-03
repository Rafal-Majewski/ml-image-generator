from typing import Sequence, Tuple
from src.modules.TrainingDataScaler import TrainingDataScaler
from src.modules.TrainingDatum import TrainingDatum
from src.modules.RegexLabelExtractor import RegexLabelExtractor
from src.modules.TrainingDataFileReader import TrainingDataFileReader
import torch
from torch import nn
from src.modules.Generator import Generator


if __name__ == "__main__":
	labels: Sequence[str] = ["apple", "banana"]
	imageSize: Tuple[int, int] = (32, 32)
	labelExtractor = RegexLabelExtractor("^.*/([a-zA-Z]+)-.*$")
	trainingDataScaler = TrainingDataScaler(imageSize)
	trainingDataFileReader = TrainingDataFileReader(
		labelExtractor=labelExtractor,
		scaler=trainingDataScaler,
		labels=labels,
	)
	trainingData: Sequence[TrainingDatum] = trainingDataFileReader.read("data")
	for trainingDatum in trainingData:
		print(trainingDatum)
	torch.manual_seed(723)
	generator = Generator(
		nn.Sequential(
			nn.Linear(2, 10),
			nn.ReLU(),
			nn.Linear(10, 2),
			nn.Tanh()
		)
	)
