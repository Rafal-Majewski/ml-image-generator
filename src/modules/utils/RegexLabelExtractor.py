from typing import Optional
from src.modules.utils.abstract.LabelExtractor import LabelExtractor
import re


class RegexLabelExtractor(LabelExtractor):
	def __init__(self, regexString: str) -> None:
		self._regex: re.Pattern = re.compile(regexString)

	@property
	def regex(self) -> re.Pattern:
		return self._regex

	def extract(self, filepath: str) -> set[str]:
		labels: list[str] = self._regex.findall(filepath)
		# if len(labels) == 0:
		# 	raise RuntimeError(f"{filepath} does not match {self._regex}")
		return set(labels)
