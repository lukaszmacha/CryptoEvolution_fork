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

    def instantiate_model(self, input_shape: tuple[int, int], output_length: int, spatial_data_shape: tuple[int, int],
                      filters_multiplier: float = 1.5, **kwargs) -> ModelAdapterBase:
        """
        Enhanced CDT 1-D CNN with improved architecture for better performance.

        Parameters:
            input_shape: Shape of the input tensor
            output_length: Number of output classes
            spatial_data_shape: Shape of spatial data component
            filters_multiplier: Factor to increase filter counts (default: 1.5)
        """

        spatial_data_rows, spatial_data_cols = spatial_data_shape
        spatial_data_length = spatial_data_rows * spatial_data_cols

        input_vector = layers.Input((1, input_shape[0]))
        reshaped_input_vector = layers.Reshape((input_shape[0],))(input_vector)
        spatial_part = layers.Lambda(lambda x: x[:, :spatial_data_length])(reshaped_input_vector)
        non_spatial_part = layers.Lambda(lambda x: x[:, spatial_data_length:])(reshaped_input_vector)
        reshaped_spatial_part = layers.Reshape((spatial_data_rows, spatial_data_cols, 1))(spatial_part)

        # Enhanced filter counts
        base_filters = int(32 * filters_multiplier)

        # First block - increased filters
        x = layers.Conv2D(filters=base_filters, kernel_size=(4, 1), activation='relu', padding='same')(reshaped_spatial_part)
        x = layers.Conv2D(filters=base_filters, kernel_size=(4, 1), activation='relu', padding='same')(x)  # Added depth
        x = layers.MaxPooling2D(pool_size=(4, 1))(x)
        x = layers.BatchNormalization()(x)

        # Second block - increased filters
        x = layers.Conv2D(filters=base_filters*2, kernel_size=(3, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=base_filters*2, kernel_size=(3, 1), activation='relu', padding='same')(x)  # Added depth
        x = layers.MaxPooling2D(pool_size=(3, 1))(x)
        x = layers.BatchNormalization()(x)

        # Third block - increased filters
        x = layers.Conv2D(filters=base_filters*4, kernel_size=(2, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=base_filters*4, kernel_size=(2, 1), activation='relu', padding='same')(x)  # Added depth
        x = layers.MaxPooling2D(pool_size=(2, 1))(x)
        x = layers.BatchNormalization()(x)

        # Add residual/skip connections for better gradient flow
        if spatial_data_rows >= 24:  # Only if we have enough spatial dimensions
            shortcut = layers.Conv2D(filters=base_filters*4, kernel_size=(1, 1), strides=(24, 1), padding='same')(reshaped_spatial_part)
            shortcut = layers.BatchNormalization()(shortcut)
            x = layers.add([x, shortcut])
            x = layers.Activation('relu')(x)

        # Flatten CNN output
        cnn_flattened = layers.Flatten()(x)

        # Process non-spatial features separately then combine
        non_spatial_processed = layers.Dense(256, activation='relu')(non_spatial_part)
        non_spatial_processed = layers.BatchNormalization()(non_spatial_processed)

        # Combine CNN output with non-spatial features
        combined = layers.Concatenate()([cnn_flattened, non_spatial_processed])

        # First fully connected layer with more units
        x = layers.Dense(1500, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)  # Slightly increased dropout

        # Second fully connected layer
        x = layers.Dense(750, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Additional layer for more capacity
        x = layers.Dense(250, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # Output softmax layer
        output = layers.Dense(output_length, activation='softmax')(x)

        # Create and return the model
        model = Model(inputs=input_vector, outputs=output)
        return TFModelAdapter(model)
