import numpy as np


class GanTrainingDatum:
	def __init__(
		self,
		*,
		discriminations: np.ndarray,
		noise: np.ndarray, # normalized: [0.0, 1.0]
	) -> None:
		self._discriminations = discriminations
		self._noise = noise

	@property
	def discriminations(self) -> np.ndarray:
		return self._discriminations

	@property
	def noise(self) -> np.ndarray:
		return self._noise

	def __str__(self) -> str:
		return f"GanTrainingDatum(discriminations={self._discriminations}, noise={self._noise})"
