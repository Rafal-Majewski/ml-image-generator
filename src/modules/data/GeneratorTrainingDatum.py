import numpy as np


class GeneratorTrainingDatum:
	def __init__(
		self,
		*,
		discriminations: list[float],
		noiseSize: int,
	) -> None:
		self._discriminations = discriminations
		self._noise = np.random.normal(0,1, noiseSize)

	@property
	def discriminations(self) -> list[float]:
		return self._discriminations

	@property
	def noise(self) -> np.ndarray:
		return self._noise

	def __str__(self) -> str:
		return f"GeneratorTrainingDatum(discriminations={self._discriminations}, noise={self._noise})"
