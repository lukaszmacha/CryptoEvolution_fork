# model/model_blue_prints/svm_blue_print.py

# global imports
import numpy as np
from sklearn import svm, tree
from sklearn.preprocessing import StandardScaler

# local imports
from source.model import BluePrintBase
from source.model import ModelAdapterBase
from source.model import SciKitLearnModelAdapter

class SVMBluePrint(BluePrintBase):
    """
    Blueprint for creating Support Vector Machine models using scikit-learn.

    This class implements a model blueprint that constructs an SVM classifier
    with configurable parameters. Unlike CNN blueprints, it doesn't require
    spatial data reshaping but accepts the parameters for interface compatibility.
    """

    def instantiate_model(self, _: tuple[int, int], output_length: int, __: tuple[int, int] = None,
                          kernel: str = 'rbf', C: float = 1.0,
                          gamma: str = 'scale', probability: bool = True,
                          class_weight: str = None) -> ModelAdapterBase:
        """
        Creates and returns an SVM classifier wrapped in a SciKitLearnModelAdapter.

        Parameters:
            input_shape (tuple[int, int]): Shape of input features (used for compatibility)
            output_length (int): Number of output classes
            spatial_data_shape (tuple[int, int]): Not used but kept for interface compatibility
            kernel (str): Kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            probability (bool): Whether to enable probability estimates
            class_weight (str): Class weight option ('balanced' or None)

        Returns:
            ModelAdapterBase: SciKitLearnModelAdapter containing the configured SVM model
        """
        # Create pipeline with preprocessing and SVM

        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.pipeline import Pipeline

        # Combine over-sampling and under-sampling
        oversampler = SMOTE(sampling_strategy='auto')
        svm_model = svm.SVC(kernel=kernel, probability=True)

        pipeline = Pipeline([
            ('oversample', oversampler),
            ('classifier', svm_model)
        ])

        return SciKitLearnModelAdapter(pipeline)
