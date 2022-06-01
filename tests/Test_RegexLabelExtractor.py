import unittest
from src.modules.RegexLabelExtractor import RegexLabelExtractor


class Test_RegexLabelExtractor(unittest.TestCase):
	def test_extract_success(self):
		extractor = RegexLabelExtractor("^([^-]+)-.*$")
		self.assertEqual(extractor.extract("test-abc"), "test")

	def test_extract_success_multiple_groups(self):
		extractor = RegexLabelExtractor("^([^-]+)-(.*)$")
		self.assertEqual(extractor.extract("test-abc"), "test")

	def test_extract_fail(self):
		extractor = RegexLabelExtractor("^([^-]+)-.*$")
		with self.assertRaises(ValueError):
			extractor.extract("test")
