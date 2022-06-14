import tensorflow.python.keras as keras


class Gan:
	def __init__(
		self,
		*,
		discriminatorModel: keras.Model,
		generatorModel: keras.Model,
	) -> None:
		super().__init__()
		self._discriminatorModel = discriminatorModel
		self._discriminatorModel.trainable = False
		self._generatorModel = generatorModel

		self._model = keras.models.Sequential()
		self._model.add(self._generatorModel)
		self._model.add(self._discriminatorModel)
		self._model.compile(
			loss=["binary_crossentropy", "binary_crossentropy"],
			optimizer="adam",
			metrics=["accuracy"],
		)
