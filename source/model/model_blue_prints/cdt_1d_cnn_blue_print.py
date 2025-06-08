# model/model_blue_prints/cdt_1d_cnn_blue_print.py

# global imports
import tensorflow as tf
from tensorflow.keras import Model, layers

# local imports
from source.model import BluePrintBase
from source.model import ModelAdapterBase
from source.model import TFModelAdapter

class CDT1DCnnBluePrint(BluePrintBase):
    """
    Blueprint for creating a CDT 1-D CNN model as described in the research.

    Architecture:
    - Three CDT 1-D convolutional and max-pooling layers
    - Two fully connected layers (1,000 and 500 units)
    - Output softmax layer

    Convolutional layers (in ascending order of depth):
    - Kernels: 4x32, 3x64, and 2x128
    - Max-pooling strides: 4, 3, and 2

    This architecture reduces the time dimension from 24 to 1 while
    increasing channel dimensions from 1 to 128.
    """

    def instantiate_model(self, input_shape: tuple[int, int], output_length: int, spatial_data_shape: tuple[int, int], **kwargs) -> ModelAdapterBase:
        """
        Creates and returns a CDT 1-D CNN model according to the specified architecture.

        Parameters:
            input_shape (tuple[int, int]): Shape of the input tensor
            output_length (int): Number of output classes
            time_steps (int): Number of time steps in the input data (default: 24)
            **kwargs: Additional parameters (not used in this blueprint)

        Returns:
            ModelAdapterBase: Model adapter wrapping the Keras model
        """

        spatial_data_rows, spatial_data_cols = spatial_data_shape
        spatial_data_length = spatial_data_rows * spatial_data_cols

        input_vector = layers.Input((1, input_shape[0]))
        reshaped_input_vector = layers.Reshape((input_shape[0],))(input_vector)
        spatial_part = layers.Lambda(lambda x: x[:, :spatial_data_length])(reshaped_input_vector)
        non_spatial_part = layers.Lambda(lambda x: x[:, spatial_data_length:])(reshaped_input_vector)
        reshaped_spatial_part = layers.Reshape((spatial_data_rows, spatial_data_cols, 1))(spatial_part)

        # First CDT 1-D convolutional layer with kernel size 4, 32 filters
        x = layers.Conv2D(filters=32, kernel_size=(4, 1), activation='relu', padding='same')(reshaped_spatial_part)
        x = layers.MaxPooling2D(pool_size=(4, 1))(x)
        x = layers.BatchNormalization()(x)

        # Second CDT 1-D convolutional layer with kernel size 3, 64 filters
        x = layers.Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(3, 1))(x)
        x = layers.BatchNormalization()(x)

        # Third CDT 1-D convolutional layer with kernel size 2, 128 filters
        x = layers.Conv2D(filters=128, kernel_size=(2, 1), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 1))(x)
        x = layers.BatchNormalization()(x)

        # Flatten the output for fully connected layers
        x = layers.Flatten()(x)

        # First fully connected layer with 1,000 units
        x = layers.Dense(1000, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Second fully connected layer with 500 units
        x = layers.Dense(500, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Output softmax layer
        output = layers.Dense(output_length, activation='softmax')(x)

        # Create and return the model
        model = Model(inputs=input_vector, outputs=output)
        return TFModelAdapter(model)
