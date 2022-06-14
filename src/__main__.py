import random
from typing import Tuple
from src.modules.data import DiscriminatorTrainingDatum
from src.modules.models.Discriminator import Discriminator
from src.modules.models.Generator import Generator
from src.modules.data.DiscriminatorTrainingDatum import TrainingDatum
from src.modules.RegexLabelExtractor import RegexLabelExtractor
from src.modules.TrainingDataFileReader import TrainingDataFileReader
import tensorflow.python.keras as keras
import tensorflow as tf
from PIL import Image as PILImage
import os
import tabulate


labels: list[str] = ["apple", "banana", "person", "orange"]
imageSize: Tuple[int, int] = (36, 36)
generatorNoiseNeuronsCount = 20

def createDiscriminatorModel() -> keras.Model:
	model = keras.models.Sequential()
	inputNeuronsCount = imageSize[0] * imageSize[1] * 3
	outputNeuronsCount = len(labels)
	model.add(keras.layers.Dense(inputNeuronsCount, activation="relu", input_shape=(inputNeuronsCount,)))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(790, activation="relu"))
	model.add(keras.layers.Dropout(0.12))
	model.add(keras.layers.Dense(450, activation="relu"))
	model.add(keras.layers.Dropout(0.08))
	model.add(keras.layers.Dense(320, activation="relu"))
	model.add(keras.layers.Dropout(0.05))
	model.add(keras.layers.Dense(210, activation="relu"))
	model.add(keras.layers.Dense(105, activation="relu"))
	model.add(keras.layers.Dense(70, activation="relu"))
	model.add(keras.layers.Dense(50, activation="relu"))
	model.add(keras.layers.Dense(20, activation="relu"))
	model.add(keras.layers.Dense(outputNeuronsCount, activation="tanh"))
	model.compile(
		loss="binary_crossentropy",
		optimizer="adam",
		metrics=["accuracy"],
	)
	return model

def createGeneratorModel() -> keras.Model:
	model = keras.models.Sequential()
	inputNeuronsCount = len(labels) + generatorNoiseNeuronsCount
	outputNeuronsCount = imageSize[0] * imageSize[1] * 3
	model.add(keras.layers.Dense(inputNeuronsCount, activation="relu", input_shape=(inputNeuronsCount,)))
	model.add(keras.layers.Dense(40, activation="relu"))
	model.add(keras.layers.Dense(90, activation="relu"))
	model.add(keras.layers.Dense(210, activation="relu"))
	model.add(keras.layers.Dense(320, activation="relu"))
	model.add(keras.layers.Dense(450, activation="relu"))
	model.add(keras.layers.Dense(790, activation="relu"))
	model.add(keras.layers.Dense(outputNeuronsCount, activation="relu"))
	model.compile(
		loss="binary_crossentropy",
		optimizer="adam",
		metrics=["accuracy"],
	)
	return model

def trainStep(
	discriminator: Discriminator,
	generator: Generator,
	realData: list[DiscriminatorTrainingDatum],
) -> Tuple[float, float]:
	discriminator.train(realData)
	generator.train(realData)
	return discriminator.getLoss(), generator.getLoss()
	

def testDiscriminator(discriminator: Discriminator) -> None:
	print("---- TESTING DISCRIMINATOR -----")
	table = []
	headers = ["filename", *labels]
	for filename in os.listdir("test_data"):
		image = PILImage.open(os.path.join("test_data", filename))
		row: list[str] = [filename]
		discriminations: list[float] = discriminator.discriminate(image)
		for i in range(len(discriminations)):
			row.append(str(round(100*discriminations[i])) + "%")
		table.append(row)
	print(tabulate.tabulate(table, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
	tf.random.set_seed(42)
	random.seed(42)

	trainingDataFileReader = TrainingDataFileReader(
		labelExtractor=RegexLabelExtractor("^training_data/([a-z A-Z]+)/.*$"),
		labels=labels,
	)
	trainingData: list[TrainingDatum] = trainingDataFileReader.read("training_data")

	discriminator = Discriminator(
		model=createDiscriminatorModel(),
		imageSize=imageSize,
	)
	# discriminator.train(trainingData, epochs=3)
	testDiscriminator(discriminator)
	generator = Generator(
		model=createGeneratorModel(),
		imageSize=imageSize,
		noiseNeuronsCount=generatorNoiseNeuronsCount,
	)
	generator.generate([1, 0, 0, 0]).show()

