import random
from statistics import mode
from typing import Tuple
import numpy as np
from src.modules.data.GanTrainingDatum import GanTrainingDatum
from src.modules.models.Discriminator import Discriminator
from src.modules.models.Gan import Gan
from src.modules.models.Generator import Generator
from src.modules.data.DiscriminatorTrainingDatum import DiscriminatorTrainingDatum
from src.modules.utils.RegexLabelExtractor import RegexLabelExtractor
from src.modules.utils.DiscriminatorTrainingDataFileReader import DiscriminatorTrainingDataFileReader
import keras
import tensorflow as tf
from PIL import Image as PILImage
import os
import tabulate


labels: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# labels: list[str] = ["apple"]
imageSize: Tuple[int, int] = (16, 16)
generatorNoiseNeuronsCount = 10

def createDiscriminatorModel() -> keras.Model:
	model = keras.models.Sequential()
	# model.trainable = False
	# inputNeuronsCount = imageSize[0] * imageSize[1] * 3
	# outputNeuronsCount = len(labels)
	# model.add(keras.layers.Conv2D(
	# 	32, kernel_size=(3, 3), strides=(2, 2), padding="same", input_shape=(imageSize[0], imageSize[1], 3)
	# ))
	model.add(keras.layers.InputLayer(input_shape=(imageSize[0], imageSize[1], 3)))
	model.add(keras.layers.Conv2D(
		160, kernel_size=(3, 3), strides=(2, 2), padding="same",
	))
	model.add(keras.layers.LeakyReLU(alpha=0.2))

	model.add(keras.layers.Conv2D(
		140, kernel_size=(3, 3), strides=(2, 2), padding="same",
	))
	model.add(keras.layers.LeakyReLU(alpha=0.2))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(
		500
	))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Dense(
		len(labels), activation="sigmoid"
	))

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
	model.add(keras.layers.Input(shape=(inputNeuronsCount,)))
	# model.add(keras.layers.Dense(70))
	# model.add(keras.layers.LeakyReLU(alpha=0.2))
	# model.add(keras.layers.Dense(140))
	# model.add(keras.layers.LeakyReLU(alpha=0.2))
	# model.add(keras.layers.Dense(480))
	# model.add(keras.layers.LeakyReLU(alpha=0.2))
	# model.add(keras.layers.Dense(600))
	# model.add(keras.layers.LeakyReLU(alpha=0.2))
	# model.add(keras.layers.Dense(1300))
	# model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Dense(1920))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Reshape((4, 4, 120)))
	model.add(keras.layers.Conv2D(
		180, kernel_size=(3, 3), strides=(1, 1), padding="same",
	))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Conv2D(
		240, kernel_size=(3, 3), strides=(1, 1), padding="same",
	))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.UpSampling2D(size=(2, 2)))
	model.add(keras.layers.Conv2D(
		280, kernel_size=(3, 3), strides=(1, 1), padding="same",
	))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.UpSampling2D(size=(2, 2)))
	model.add(keras.layers.Conv2D(
		3, kernel_size=(3, 3), strides=(1, 1), padding="same"
	))
	model.add(keras.layers.Activation("sigmoid"))
	return model


def generateFakeDiscriminatorTrainingDatum(
	generator: Generator,
) -> DiscriminatorTrainingDatum:
	noise = np.random.normal(0, 1, size=generatorNoiseNeuronsCount)
	discriminations = np.zeros(len(labels))
	labelId: int = random.randint(0, len(labels) - 1)
	discriminations[labelId] = random.uniform(0.9, 1.0)
	image = generator.generate(discriminations, noise)
	discriminations[labelId] = 0
	return DiscriminatorTrainingDatum(
		discriminations=discriminations,
		pixels=np.array(image.resize(imageSize)),
	)

def generateDiscriminatorTrainingData(
	inputData: list[DiscriminatorTrainingDatum],
	generator: Generator,
	size: int,
) -> list[DiscriminatorTrainingDatum]:
	generatedData: list[DiscriminatorTrainingDatum] = []
	for _ in range(size):
		if random.random() < 0.9:
			generatedData.append(random.choice(inputData))
		else:
			generatedData.append(generateFakeDiscriminatorTrainingDatum(generator))
	return generatedData

def generateGanTrainingData(
	size: int,
) -> list[DiscriminatorTrainingDatum]:
	generatedData: list[DiscriminatorTrainingDatum] = []
	for _ in range(size):
		discriminations = np.zeros(len(labels))
		labelId: int = random.randint(0, len(labels) - 1)
		discriminations[labelId] = random.uniform(0.9, 1.0)
		noise = np.random.normal(0, 1, size=generatorNoiseNeuronsCount)
		generatedData.append(GanTrainingDatum(
			discriminations=discriminations,
			noise=noise,
		))
	return generatedData

constnoise = np.random.normal(0, 1, size=generatorNoiseNeuronsCount)
constdiscrims = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
constimgs = []

def train(
	gan: Gan,
	inputData: list[DiscriminatorTrainingDatum],
) -> None:
	for epochNumber in range(800):
		dtd = generateDiscriminatorTrainingData(inputData, gan.generator, 12)
		gan.discriminator.model.trainable = True
		dloss, dacc = gan.discriminator.train(dtd)

		gtd = generateGanTrainingData(14)
		gan.discriminator.model.trainable = False
		gloss, gacc = gan.train(gtd)
		print(epochNumber, "dloss:", dloss, "dacc:", dacc, "gloss:", gloss, "gacc:", gacc)
		img = gan.generator.generate(
			constdiscrims,
			constnoise,
		)
		constimgs.append(img)
		
		# # if epochNumber % 20 == 0:
		# # 	img.show()
		img = constimgs[0]
		img.save(
			"gan_const.gif",
			format="GIF",
			append_images=constimgs[1:],
			save_all=True,
			duration=len(constimgs) * 0.1,
			loop=0,
		)

def test(
	gan: Gan,
) -> None:
	discriminator = gan.discriminator
	generator = gan.generator
	# read dir test_data for files
	filenames = os.listdir("test_data/num")
	for filename in filenames:
		image = PILImage.open("test_data/num/" + filename)
		print(filename, discriminator.discriminate(image))

	# for datum in testData:
	# 	discriminations = datum.discriminations
	# 	pixels = datum.pixels
	# 	print(
	# 		discriminations, discriminator.discriminate(
	# 			generator.numersToImage(pixels),
	# 		),
	# 	)


if __name__ == "__main__":
	tf.random.set_seed(42)
	random.seed(42)


	discriminatorTrainingDataFileReader = DiscriminatorTrainingDataFileReader(
		labelExtractor=RegexLabelExtractor("^training_data/([0-9])/.*$"),
		labels=labels,
		imageSize=imageSize,
	)
	inputData: list[DiscriminatorTrainingDatum] = discriminatorTrainingDataFileReader.read("training_data")

	discriminator = Discriminator(
		model=createDiscriminatorModel(),
		imageSize=imageSize,
	)
	generator = Generator(
		model=createGeneratorModel(),
		imageSize=imageSize,
	)
	gan = Gan(
		discriminator=discriminator,
		generator=generator,
	)
	train(gan, inputData)
	test(gan)
