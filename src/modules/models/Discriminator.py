from typing import Tuple
import keras
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
		self._model.compile(
			loss="binary_crossentropy",
			optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
			metrics=["accuracy"],
		)
		self._imageSize = imageSize

	@property
	def model(self) -> keras.Model:
		return self._model

	def imageToNumbers(self, image: PILImage.Image) -> np.ndarray:
		return np.array(
			image.resize(self._imageSize).convert("RGB"),
		) / 255

	def discriminate(self, image: PILImage.Image) -> np.ndarray:
		discriminations: np.ndarray = self._model(
			np.array([self.imageToNumbers(image)]),
			training=False,
		).numpy()[0]
		return discriminations

	def train(
		self,
		data: list[DiscriminatorTrainingDatum],
	):
		x: np.ndarray = np.array(
			[datum.pixels for datum in data]
		)
		y: np.ndarray = np.array(
			[datum.discriminations for datum in data]
		)

		return self._model.train_on_batch(x, y)
