from src.modules.data.DiscriminatorTrainingDatum import DiscriminatorTrainingDatum
from src.modules.utils.abstract.LabelExtractor import LabelExtractor
from PIL import Image as PILImage
import os
import numpy as np
from typing import Tuple


class DiscriminatorTrainingDataFileReader:
	def __init__(self,
		labelExtractor: LabelExtractor,
		labels: list[str],
		imageSize: Tuple[int, int],
	) -> None:
		self._labelExtractor = labelExtractor
		self._labels = labels
		self._imageSize = imageSize

	def _readDatum(self, filepath: str) -> DiscriminatorTrainingDatum:
		datumLabels: set[str] = self._labelExtractor.extract(filepath)
		return DiscriminatorTrainingDatum(
			discriminations=[1.0 if label in datumLabels else 0.0 for label in self._labels],
			pixels=np.array(
				PILImage.open(filepath)
				.resize(self._imageSize)
				.convert("RGB")
			) / 255,
		)

	def read(self, dirpath: str) -> list[DiscriminatorTrainingDatum]:
		trainingData = list[DiscriminatorTrainingDatum]()
		for root, dirs, files in os.walk(dirpath):
			for file in files:
				trainingData.append(self._readDatum(os.path.join(root, file)))
		return trainingData
