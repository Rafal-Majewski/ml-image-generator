from typing import Tuple
import tensorflow.python.keras as keras
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

	def numbersToImage(self, numbers: np.ndarray) -> PILImage:
		return PILImage.fromarray(
			(numbers.reshape(self._imageSize[0], self._imageSize[1], 3) * 255).astype(np.uint8),
		).convert("RGB")

	def generate(self, discriminations: list[float], noise: list[float]) -> PILImage:
		generatedNumbers: np.ndarray = self._model.predict(
			np.array([discriminations + noise]),
			training=False,
		).numpy()[0]
		return self.numbersToImage(generatedNumbers)
