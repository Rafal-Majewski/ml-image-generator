from PIL import Image as PILImage
import numpy as np


class DiscriminatorTrainingDatum:
	def __init__(
		self,
		*,
		discriminations: np.ndarray,
		image: PILImage.Image
	) -> None:
		self._discriminations = discriminations
		self._image = image

	@property
	def discriminations(self) -> np.ndarray:
		return self._discriminations

	@property
	def image(self) -> PILImage.Image:
		return self._image

	def __str__(self) -> str:
		return f"DiscriminatorTrainingDatum(discriminations={self._discriminations}, image={self._image})"
