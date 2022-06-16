import io
import pathlib
import random
from typing import Tuple
from matplotlib import pyplot
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
from datetime import datetime as Datetime


labels: list[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
imageSize: Tuple[int, int] = (28, 28)
generatorNoiseNeuronsCount = 20
batchSize = 60
epochsCount = 1200000
realDatumWeight = 0.5
labelRegex="(?<=training_data\\/)[0-9](?=\\/)"

def createDiscriminatorModel() -> keras.Model:
	model = keras.models.Sequential()
	model.add(keras.layers.InputLayer(input_shape=(imageSize[0], imageSize[1], 3)))
	model.add(keras.layers.Conv2D(30, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Conv2D(30, (3, 3), strides=(2, 2), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Conv2D(30, (3, 3), strides=(2, 2), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(len(labels)))
	model.add(keras.layers.Activation("sigmoid"))
	return model

def createGeneratorModel() -> keras.Model:
	model = keras.models.Sequential()
	inputNeuronsCount = len(labels) + generatorNoiseNeuronsCount
	model.add(keras.layers.Input(shape=(inputNeuronsCount,)))
	model.add(keras.layers.Dense(40))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Dense(60))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Dense(49))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Reshape((7, 7, 1)))
	model.add(keras.layers.Conv2D(30, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.UpSampling2D((2, 2)))
	model.add(keras.layers.Conv2D(40, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.UpSampling2D((2, 2)))
	model.add(keras.layers.Conv2D(60, (3, 3), padding="same"))
	model.add(keras.layers.LeakyReLU(alpha=0.2))
	model.add(keras.layers.Conv2D(3, (3, 3), padding="same"))
	model.add(keras.layers.Activation("sigmoid"))
	return model

def generateRandomNoise() -> np.ndarray:
	return np.random.normal(0, 1, size=generatorNoiseNeuronsCount)

runName = Datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def generateFakeDiscriminatorTrainingDatum(
	generator: Generator,
) -> DiscriminatorTrainingDatum:
	discriminations = generateRandomDiscriminations()
	noise = generateRandomNoise()
	image = generator.generate(discriminations, noise)
	discriminations = np.array([0 for _ in discriminations])
	return DiscriminatorTrainingDatum(
		discriminations=discriminations,
		pixels=np.array(image),
	)

def generateDiscriminatorTrainingData(
	realData: list[DiscriminatorTrainingDatum],
	generator: Generator,
	size: int,
) -> list[DiscriminatorTrainingDatum]:
	generatedData: list[DiscriminatorTrainingDatum] = []
	for _ in range(round(realDatumWeight * size)):
		generatedData.append(random.choice(realData))
	for _ in range(size - len(generatedData)):
		generatedData.append(generateFakeDiscriminatorTrainingDatum(generator))
	random.shuffle(generatedData)
	return generatedData

def generateRandomGanTrainingDatum() -> GanTrainingDatum:
	discriminations = generateRandomDiscriminations()
	noise = generateRandomNoise()
	return GanTrainingDatum(
		discriminations=discriminations,
		noise=noise,
	)

def generateGanTrainingData(
	size: int,
) -> list[DiscriminatorTrainingDatum]:
	generatedData: list[DiscriminatorTrainingDatum] = []
	for _ in range(size):
		generatedData.append(generateRandomGanTrainingDatum())
	return generatedData

def train(
	gan: Gan,
	realData: list[DiscriminatorTrainingDatum],
) -> None:
	for epochNumber in range(epochsCount):
		if epochNumber % 5 == 0:
			for i, ganDatum in enumerate(constGanData):
				pyplot.subplot(5, 5, i + 1)
				image = gan.generator.generate(ganDatum.discriminations, ganDatum.noise)
				pyplot.imshow(image)
				pyplot.axis("off")
			imgBuf = io.BytesIO()
			pyplot.savefig(imgBuf, cmap="gray_r", format="png")
			pyplot.close()
			plotImg = PILImage.open(imgBuf)
			constImgs.append(plotImg)
			pathlib.Path(f"output_data/{runName}/gifs").mkdir(parents=True, exist_ok=True)
			constImgs[0].save(
				f"output_data/{runName}/gifs/{epochNumber}.gif",
				save_all=True,
				append_images=constImgs[1:],
				loop=0,
			)
			pathlib.Path(f"output_data/{runName}/models/{epochNumber}").mkdir(parents=True, exist_ok=True)
			gan.generator.model.save(
				f"output_data/{runName}/models/{epochNumber}/generator.h5"
			)	
			gan.discriminator.model.save(
				f"output_data/{runName}/models/{epochNumber}/discriminator.h5"
			)
		dtd = generateDiscriminatorTrainingData(realData, gan.generator, batchSize)
		gan.discriminator.model.trainable = True
		dloss, dacc = gan.discriminator.train(dtd)

		gtd = generateGanTrainingData(batchSize)
		gan.discriminator.model.trainable = False
		gloss, gacc = gan.train(gtd)
		print(epochNumber, "dloss:", dloss, "dacc:", dacc, "gloss:", gloss, "gacc:", gacc)

def generateRandomDiscriminations() -> np.ndarray:
	discriminations = np.zeros(len(labels))
	labelId: int = random.randint(0, len(labels) - 1)
	discriminations[labelId] = random.uniform(0.9, 1.0)
	return discriminations

def generateSample(generator: Generator) -> PILImage:
	discriminations = generateRandomDiscriminations()
	noise = generateRandomNoise()
	image = generator.generate(discriminations, noise)
	return image

def generateSamples(
	generator: Generator,
	count: int,
) -> list[PILImage.Image]:
	images = []
	for _ in range(count):
		images.append(generateSample(generator))
	return images

constGanData = [GanTrainingDatum(
	discriminations=generateRandomDiscriminations(),
	noise=generateRandomNoise(),
) for _ in range(25)]
constImgs: list[PILImage.Image] = []

def test(
	gan: Gan,
) -> None:
	generator = gan.generator
	discriminator = gan.discriminator
	# for sample in generateSamples(generator, 10):
	# 	sample.show()

if __name__ == "__main__":
	tf.random.set_seed(42)
	random.seed(42)


	discriminatorTrainingDataFileReader = DiscriminatorTrainingDataFileReader(
		# labelExtractor=RegexLabelExtractor("(?<=[-\/])[a-z]+(?=[-\.])"),
		labelExtractor=RegexLabelExtractor(labelRegex),
		labels=labels,
		imageSize=imageSize,
	)
	realData: list[DiscriminatorTrainingDatum] = discriminatorTrainingDataFileReader.read("training_data")

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
	train(gan, realData)
	test(gan)
