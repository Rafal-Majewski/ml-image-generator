import PIL.Image as PILImage


class TrainingDatum:
	def __init__(
		self,
		*,
		label: str,
		image: PILImage.Image
	) -> None:
		self._label = label
		self._image = image

	@property
	def label(self) -> str:
		return self._label

	@property
	def image(self) -> PILImage.Image:
		return self._image
