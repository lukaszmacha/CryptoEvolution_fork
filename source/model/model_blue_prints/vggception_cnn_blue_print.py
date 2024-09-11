# model/model_blue_prints/basic_cnn_blue_print.py

from tensorflow.keras import Model, layers
import math

from .base_blue_print import BaseBluePrint
from ..model_building_blocks.vgg16_block import Vgg16Block
from ..model_building_blocks.xception_block import XceptionBlock

class VGGceptionCnnBluePrint(BaseBluePrint):
    """
    """

    def instantiate_model(self, input_length: int, output_length: int, spatial_data_shape: tuple[int, int],
                          number_of_filters: int = 32, cnn_squeezing_coeff: int = 2, dense_squeezing_coeff: int = 2,
                          dense_repetition_coeff: int = 1, filters_number_coeff: int = 2) -> Model:
        """
        """

        spatial_data_rows, spatial_data_cols = spatial_data_shape
        spatial_data_length = spatial_data_rows * spatial_data_cols 

        input_vector = layers.Input((input_length,))
        spatial_part = layers.Lambda(lambda x: x[:, :spatial_data_length])(input_vector)
        non_spatial_part = layers.Lambda(lambda x: x[:, spatial_data_length:])(input_vector)
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
                                                      int(math.log(concatenated_parts.shape[-1], 2))))
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

        return Model(inputs=input_vector, outputs=output)