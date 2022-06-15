from typing import Tuple
import keras
from PIL import Image as PILImage
import numpy as np


class Generator:
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

	def numbersToImage(self, numbers: np.ndarray) -> PILImage.Image:
		return PILImage.fromarray(
			(numbers * 255).astype(np.uint8),
			"RGB",
		)

	def generate(self, discriminations: np.ndarray, noise: np.ndarray) -> PILImage.Image:
		generatedNumbers: np.ndarray = self._model(
			np.array([np.concatenate((discriminations, noise), axis=None)]),
			training=False,
		).numpy()[0]
		return self.numbersToImage(generatedNumbers)
