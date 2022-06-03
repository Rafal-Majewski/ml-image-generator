from typing import Sequence
from src.modules.TrainingDatum import TrainingDatum
from src.modules.abstract.LabelExtractor import LabelExtractor
from PIL import Image as PILImage
from src.modules.TrainingDataScaler import TrainingDataScaler
import os


class TrainingDataFileReader:
	def __init__(self,
		labelExtractor: LabelExtractor,
		scaler: TrainingDataScaler,
		labels: Sequence[str],
	) -> None:
		self._labelExtractor = labelExtractor
		self._scaler = scaler
		self._labels = labels
		self._labelsIds = {label: i for i, label in enumerate(labels)}

	def _assertValidLabel(self, label: str) -> None:
		if label not in self._labels:
			raise RuntimeError("Unknown label: " + label)

	def _readDatum(self, filepath: str) -> TrainingDatum:
		labels = self._labelExtractor.extract(filepath)
		for label in labels:
			self._assertValidLabel(label)
		
		return TrainingDatum(
			labelsIds=[self._labelsIds[label] for label in labels],
			image=self._scaler.scale(PILImage.open(filepath)),
		)

	def read(self, dirpath: str) -> list[TrainingDatum]:
		trainingData = list[TrainingDatum]()
		for filename in os.listdir(dirpath):
			trainingDatum = self._readDatum(dirpath + "/" + filename)
			trainingData.append(trainingDatum)
		return trainingData
