from torch import nn
from PIL import Image as PILImage


class Generator(nn.Module):
	def __init__(self, model: nn.Module):
		super().__init__()
		self.model = model

	def forward(self, x):
		return self.model(x)
