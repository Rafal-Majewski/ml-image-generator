from src.modules.data.DiscriminatorTrainingDatum import DiscriminatorTrainingDatum
from src.modules.utils.abstract.LabelExtractor import LabelExtractor
from PIL import Image as PILImage
import os


class TrainingDataFileReader:
	def __init__(self,
		labelExtractor: LabelExtractor,
		labels: list[str],
	) -> None:
		self._labelExtractor = labelExtractor
		self._labels = labels

	def _readDatum(self, filepath: str) -> DiscriminatorTrainingDatum:
		datumLabels: set[str] = self._labelExtractor.extract(filepath)
		
		return DiscriminatorTrainingDatum(
			discriminations=[1.0 if label in datumLabels else 0.0 for label in self._labels],
			image=PILImage.open(filepath),
		)

	def read(self, dirpath: str) -> list[DiscriminatorTrainingDatum]:
		trainingData = list[DiscriminatorTrainingDatum]()
		for root, dirs, files in os.walk(dirpath):
			for file in files:
				trainingData.append(self._readDatum(os.path.join(root, file)))
		return trainingData
