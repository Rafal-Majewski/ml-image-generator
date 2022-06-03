from typing import Tuple
from PIL import Image as PILImage


class TrainingDataScaler:
	def __init__(self, targetImageSize: Tuple[int, int]) -> None:
		self._targetImageSize = targetImageSize

	@property
	def targetImageSize(self) -> Tuple[int, int]:
		return self._targetImageSize

	def scale(self, image: PILImage.Image) -> PILImage.Image:
		return image.resize(self.targetImageSize)
