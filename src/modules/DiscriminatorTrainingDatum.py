from PIL import Image as PILImage


class TrainingDatum:
	def __init__(
		self,
		*,
		discriminations: list[float],
		image: PILImage.Image
	) -> None:
		self._discriminations = discriminations
		self._image = image

	@property
	def discriminations(self) -> list[float]:
		return self._discriminations

	@property
	def image(self) -> PILImage.Image:
		return self._image

	def __str__(self) -> str:
		return f"TrainingDatum(discriminations={self._discriminations}, image={self._image})"
