# agent/strategies/classification_learning_agent.py

# global imports
import numpy as np
import pandas as pd
from typing import Any
from tensorflow.keras.callbacks import Callback

# local imports
from source.agent import AgentBase
from source.agent import ClassificationTestable

class ClassificationLearningAgent(AgentBase, ClassificationTestable):
    """"""

    def classification_fit(self, input_data, output_data, batch_size: int,
                           epochs: int, callbacks: list[Callback]) -> dict[str, Any]:
        """"""

        return self._model_adapter.fit(input_data, output_data, batch_size = batch_size,
                                       epochs = epochs, callbacks = callbacks).history

    def classify(self, data: pd.DataFrame) -> list[list[float]]:
        """"""

        return self._model_adapter.predict(data)
