import random
from typing import Tuple
import numpy as np
from src.modules.data.GanTrainingDatum import GanTrainingDatum
from src.modules.models.Discriminator import Discriminator
from src.modules.models.Gan import Gan
from src.modules.models.Generator import Generator
from src.modules.data.DiscriminatorTrainingDatum import DiscriminatorTrainingDatum
from src.modules.utils.RegexLabelExtractor import RegexLabelExtractor
from src.modules.utils.DiscriminatorTrainingDataFileReader import DiscriminatorTrainingDataFileReader
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
	model.trainable = False
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

def generateFakeDiscriminatorTrainingDatum(
	generator: Generator,
) -> DiscriminatorTrainingDatum:
	noise = np.random.uniform(-1, 1, size=generatorNoiseNeuronsCount)
	labelId: int = random.randint(0, len(labels) - 1)
	discriminations = [0 for _ in range(len(labels))]
	discriminations[labelId] = 1
	image = generator.generate(discriminations, noise)
	discriminations[labelId] = -1
	return DiscriminatorTrainingDatum(
		discriminations=discriminations,
		image=image,
	)

def generateDiscriminatorTrainingData(
	inputData: list[DiscriminatorTrainingDatum],
	generator: Generator,
	size: int,
) -> list[DiscriminatorTrainingDatum]:
	generatedData: list[DiscriminatorTrainingDatum] = []
	for _ in range(size):
		if random.random() < 0.5:
			generatedData.append(generateFakeDiscriminatorTrainingDatum(generator))
		else:
			generatedData.append(random.choice(inputData))
	return generatedData

def generateGanTrainingData(
	size: int,
) -> list[DiscriminatorTrainingDatum]:
	generatedData: list[DiscriminatorTrainingDatum] = []
	for _ in range(size):
		discriminations = [0 for _ in range(len(labels))]
		labelId: int = random.randint(0, len(labels) - 1)
		discriminations[labelId] = 1
		noise = np.random.normal(0, 1, size=generatorNoiseNeuronsCount)
		generatedData.append(GanTrainingDatum(
			discriminations=discriminations,
			noise=noise,
		))
	return generatedData

print(generateGanTrainingData(10)[0])

# def train(
# 	gan: Gan,
# 	inputData: list[DiscriminatorTrainingDatum],
# ) -> None:
# 	for epochNumber in range(5):
# 		dtd = generateDiscriminatorTrainingData(inputData, gan.generator, 10)
# 		gan.discriminator.train(dtd)



# if __name__ == "__main__":
# 	tf.random.set_seed(42)
# 	random.seed(42)

# 	trainingDataFileReader = DiscriminatorTrainingDataFileReader(
# 		labelExtractor=RegexLabelExtractor("^training_data/([a-z A-Z]+)/.*$"),
# 		labels=labels,
# 	)
# 	inputData: list[DiscriminatorTrainingDatum] = trainingDataFileReader.read("training_data")

# 	discriminator = Discriminator(
# 		model=createDiscriminatorModel(),
# 		imageSize=imageSize,
# 	)
# 	generator = Generator(
# 		model=createGeneratorModel(),
# 		imageSize=imageSize,
# 	)
# 	gan = Gan(
# 		discriminator=discriminator,
# 		generator=generator,
# 	)
# 	train(gan, inputData)
