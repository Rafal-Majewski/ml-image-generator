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

	def discriminate(self, image: PILImage) -> list[float]:
		discriminations: list[float] = self._model(
			np.array([self.imageToNumbers(image)]),
			training=False,
		).numpy()[0]
		return discriminations

	def train(
		self,
		data: list[DiscriminatorTrainingDatum],
	) -> None:
		x: np.ndarray = np.array(
			[self.imageToNumbers(datum.image) for datum in data]
		)
		y: np.ndarray = np.array(
			[datum.discriminations for datum in data]
		)

		self._model.train_on_batch(x, y)