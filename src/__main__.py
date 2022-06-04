import random
from typing import Sequence, Tuple
from src.modules.Discriminator import Discriminator
from src.modules.TrainingDatum import TrainingDatum
from src.modules.RegexLabelExtractor import RegexLabelExtractor
from src.modules.TrainingDataFileReader import TrainingDataFileReader
import tensorflow.python.keras as keras
import tensorflow as tf
from PIL import Image as PILImage


def createDiscriminatorModel(imageSize: Tuple[int, int], labels: Sequence[str]) -> keras.Model:
	model = keras.models.Sequential()
	inputNeuronsCount = imageSize[0] * imageSize[1] * 3
	outputNeuronsCount = len(labels)
	model.add(keras.layers.Dense(inputNeuronsCount, activation="relu", input_shape=(inputNeuronsCount,)))
	model.add(keras.layers.Dense((inputNeuronsCount + outputNeuronsCount) // 2, activation="relu"))
	model.add(keras.layers.Dense(len(labels), activation="sigmoid"))
	model.compile(
		loss="binary_crossentropy",
		optimizer="adam",
		metrics=["accuracy"],
	)
	return model


if __name__ == "__main__":
	tf.random.set_seed(42)
	random.seed(42)
	labels: list[str] = ["apple", "banana"]
	imageSize: Tuple[int, int] = (64, 64)
	trainingDataFileReader = TrainingDataFileReader(
		labelExtractor=RegexLabelExtractor("^data/([a-zA-Z]+)/.*$"),
		labels=labels,
	)
	trainingData: list[TrainingDatum] = trainingDataFileReader.read("data")

	discriminator = Discriminator(
		model=createDiscriminatorModel(imageSize, labels),
		imageSize=imageSize,
	)
	# discriminator.train(
	# 	trainingData,
	# 	epochs=3,
	# )
	# testImg: PILImage = PILImage.open("kijow.png")
	# print(
	# 	"kijow.png",
	# 	discriminator.discriminate(testImg),
	# )
