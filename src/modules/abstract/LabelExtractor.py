from abc import ABC, abstractmethod


class LabelExtractor(ABC):
	@abstractmethod
	def extract(self, filename: str) -> str:
		pass
