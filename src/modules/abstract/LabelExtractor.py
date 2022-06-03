from abc import ABC, abstractmethod


class LabelExtractor(ABC):
	@abstractmethod
	def extract(self, filepath: str) -> list[str]:
		pass
