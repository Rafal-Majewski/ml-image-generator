import random
from typing import Tuple
import tensorflow.python.keras as keras
from PIL import Image as PILImage
from src.modules.data.DiscriminatorTrainingDatum import DiscriminatorTrainingDatum
import numpy as np


class Discriminator:
	def __init__(
		self,
		*,
		model: keras.Model,
		imageSize: Tuple[int, int],
	) -> None:
		super().__init__()
		self._model = model
		self._imageSize = imageSize

	@property
	def model(self) -> keras.Model:
		return self._model

	def imageToNumbers(self, image: PILImage) -> np.ndarray:
		return np.array(
			image.convert("RGB").resize(self._imageSize).getdata()
		).flatten() / 255

	def discriminate(self, image: PILImage) -> np.ndarray:
		discriminations: np.ndarray = self._model.predict(
			np.array([self.imageToNumbers(image)]),
		)[0]
		return discriminations

	def train(
		self,
		data: list[DiscriminatorTrainingDatum],
	):
		x: np.ndarray = np.array(
			[self.imageToNumbers(datum.image) for datum in data]
		)
		y: np.ndarray = np.array(
			[datum.discriminations for datum in data]
		)

		return self._model.train_on_batch(x, y)
