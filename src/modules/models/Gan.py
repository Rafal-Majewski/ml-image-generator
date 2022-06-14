import numpy as np
import tensorflow.python.keras as keras

from src.modules.data.GanTrainingDatum import GanTrainingDatum


class Gan:
	def __init__(
		self,
		*,
		discriminatorModel: keras.Model,
		generatorModel: keras.Model,
	) -> None:
		super().__init__()
		self._discriminatorModel = discriminatorModel
		self._generatorModel = generatorModel

		self._model = keras.models.Sequential()
		self._model.add(self._generatorModel)
		self._model.add(self._discriminatorModel)
		self._model.compile(
			loss=["binary_crossentropy", "binary_crossentropy"],
			optimizer="adam",
			metrics=["accuracy"],
		)

	def train(
		self,
		data: list[GanTrainingDatum],
	) -> None:
		x: np.ndarray = np.array(
			[datum.discriminations + datum.noise for datum in data]
		)
		y: np.ndarray = np.array(
			[datum.discriminations for datum in data]
		)

		self._model.train_on_batch(x, y)
