from PIL import Image as PILImage
import numpy as np


class DiscriminatorTrainingDatum:
	def __init__(
		self,
		*,
		discriminations: np.ndarray,
		pixels: np.ndarray  # normalized: [0.0, 1.0]
	) -> None:
		self._discriminations = discriminations
		self._pixels = pixels

	@property
	def discriminations(self) -> np.ndarray:
		return self._discriminations

	@property
	def pixels(self) -> np.ndarray:
		return self._pixels

	def __str__(self) -> str:
		return f"DiscriminatorTrainingDatum(discriminations={self._discriminations}, pixels={self._pixels})"
