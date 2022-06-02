from typing import Sequence
from src.modules.TrainingDatum import TrainingDatum
from src.modules.abstract.LabelExtractor import LabelExtractor
from PIL import Image as PILImage
import os


class TrainingDataFileReader:
	def __init__(self, labelExtractor: LabelExtractor) -> None:
		self._labelExtractor = labelExtractor

	@property
	def labelExtractor(self) -> LabelExtractor:
		return self._labelExtractor

	def _readDatum(self, filepath: str) -> TrainingDatum:
		return TrainingDatum(
			label=self.labelExtractor.extract(filepath),
			image=PILImage.open(filepath)
		)

	def read(self, dirpath: str) -> Sequence[TrainingDatum]:
		trainingData = list[TrainingDatum]()
		for filename in os.listdir(dirpath):
			trainingDatum = self._readDatum(dirpath + "/" + filename)
			trainingData.append(trainingDatum)
		return trainingData
