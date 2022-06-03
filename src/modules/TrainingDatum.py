from PIL import Image as PILImage


class TrainingDatum:
	def __init__(
		self,
		*,
		labelsIds: list[int],
		image: PILImage.Image
	) -> None:
		self._labelsIds = labelsIds
		self._image = image

	@property
	def labelsIds(self) -> list[int]:
		return self._labelsIds

	@property
	def image(self) -> PILImage.Image:
		return self._image

	def __str__(self) -> str:
		return f"TrainingDatum(labelsIds={self._labelsIds}, image={self._image})"
