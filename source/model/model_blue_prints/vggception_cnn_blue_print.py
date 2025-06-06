# model/model_blue_prints/basic_cnn_blue_print.py

# global imports
import math
from tensorflow.keras import Model, layers

# local imports
from source.model import BluePrintBase
from source.model import Vgg16Block
from source.model import XceptionBlock
from source.model import ModelAdapterBase
from source.model import TFModelAdapter

class VGGceptionCnnBluePrint(BluePrintBase):
    """
    Blueprint for creating a hybrid CNN architecture combining VGG and Xception patterns.

    This class implements a model blueprint that constructs a neural network with a
    combined architecture inspired by VGG16 and Xception networks. It's designed to
    process both spatial and non-spatial features by separating the input vector
    and processing them through different network components before combining them.
    """

    def instantiate_model(self, input_shape: tuple[int, int], output_length: int, spatial_data_shape: tuple[int, int],
                          number_of_filters: int = 32, cnn_squeezing_coeff: int = 2, dense_squeezing_coeff: int = 2,
                          dense_repetition_coeff: int = 1, filters_number_coeff: int = 2) -> ModelAdapterBase:
        """
        Creates and returns a hybrid VGG-Xception CNN model according to specified parameters.

        The method constructs a neural network that:
        1. Separates the input into spatial and non-spatial components
        2. Processes the spatial data through VGG16 and Xception blocks
        3. Flattens the CNN output and concatenates with non-spatial features
        4. Passes the combined features through a series of dense layers
        5. Produces a softmax output for classification

        Parameters:
            input_shape (tuple[int, int]): Shape of the input tensor
            output_length (int): Number of output classes/actions
            spatial_data_shape (tuple[int, int]): Rows and columns to reshape spatial data
            number_of_filters (int): Initial number of convolutional filters
            cnn_squeezing_coeff (int): Factor by which CNN dimensions are reduced
            dense_squeezing_coeff (int): Factor by which dense layer sizes are reduced
            dense_repetition_coeff (int): Number of dense layers of the same size to use
            filters_number_coeff (int): Factor by which filter count increases in convolutional layers

        Returns:
            Model: Keras model implementing the hybrid VGG-Xception architecture to be compiled further.
        """

        spatial_data_rows, spatial_data_cols = spatial_data_shape
        spatial_data_length = spatial_data_rows * spatial_data_cols

        input_vector = layers.Input((1, input_shape[0]))
        reshaped_input_vector = layers.Reshape((input_shape[0],))(input_vector)
        spatial_part = layers.Lambda(lambda x: x[:, :spatial_data_length])(reshaped_input_vector)
        non_spatial_part = layers.Lambda(lambda x: x[:, spatial_data_length:])(reshaped_input_vector)
        reshaped_spatial_part = layers.Reshape((spatial_data_rows, spatial_data_cols, 1))(spatial_part)

        cnn_part = Vgg16Block([(3, 1), (3, 1), (2, 1)],
                              [number_of_filters, number_of_filters])(reshaped_spatial_part)
        cnn_part = layers.BatchNormalization()(cnn_part)

        nr_of_xceptions_blocks = int(math.ceil(math.log(spatial_data_rows // 2, cnn_squeezing_coeff)))
        for _ in range(nr_of_xceptions_blocks):
            number_of_filters *= filters_number_coeff
            cnn_part = XceptionBlock([(3, 1), (3, 1), (3, 1), (1, 1)],
                                    [number_of_filters, number_of_filters, number_of_filters],
                                    [(cnn_squeezing_coeff, 1), (cnn_squeezing_coeff, 1)])(cnn_part)
            cnn_part = layers.BatchNormalization()(cnn_part)

        flatten_cnn_part = layers.Flatten()(cnn_part)
        concatenated_parts = layers.Concatenate()([flatten_cnn_part, non_spatial_part])

        closest_smaller_power_of_coeff = int(math.pow(dense_squeezing_coeff,
                                                      int(math.log(concatenated_parts.shape[-1],
                                                                   dense_squeezing_coeff))))
        dense = layers.Dense(closest_smaller_power_of_coeff, activation='relu')(concatenated_parts)
        dense = layers.BatchNormalization()(dense)

        number_of_nodes = closest_smaller_power_of_coeff // dense_squeezing_coeff
        nr_of_dense_layers = int(math.log(closest_smaller_power_of_coeff, dense_squeezing_coeff))
        for _ in range(nr_of_dense_layers):
            for _ in range(dense_repetition_coeff):
                dense = layers.Dense(number_of_nodes, activation='relu')(dense)
            dense = layers.BatchNormalization()(dense)
            number_of_nodes //= dense_squeezing_coeff
            if int(math.log(number_of_nodes, 10)) == int(math.log(output_length, 10)) + 1:
                dense = layers.Dropout(0.3)(dense)
            elif int(math.log(number_of_nodes, 10)) == int(math.log(output_length, 10)):
                break

        output = layers.Dense(output_length, activation='softmax')(dense)

        return TFModelAdapter(Model(inputs = input_vector, outputs = output))