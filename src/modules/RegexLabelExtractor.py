from typing import Optional
from src.modules.abstract.LabelExtractor import LabelExtractor
import re


class RegexLabelExtractor(LabelExtractor):
	def __init__(self, regexString: str) -> None:
		self._regex: re.Pattern = re.compile(regexString)

	@property
	def regex(self) -> re.Pattern:
		return self._regex

	def extract(self, string: str) -> str:
		match: Optional[re.Match] = self._regex.match(string)
		if match is None:
			raise ValueError(f"{string} does not match {self._regex}")
		return match.group(1)
