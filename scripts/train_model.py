# scripts/train_model.py

import logging
import json
import os
import argparse
import urllib.parse
from typing import Any, Type
from datetime import datetime

from source.training import TrainingHandler, TrainingConfig
from source.paperspace import GradientHandler
from source.utils import CallbackFromStringConverter, ValidatorFromStringConverter, \
    ModelBluePrintFromStringConverter, OptimizerFromStringConverter, PolicyFromStringConverter, \
    LearningStrategyHandlerFromStringConverter, TestingStrategyHandlerFromStringConverter, \
    LabelAnnotatorFromStringConverter
from source.aws import AWSHandler

CONVERTER_TYPE_MAP: dict[str, Type[Any]] = {
    'model_blue_print': ModelBluePrintFromStringConverter,
    'validator': ValidatorFromStringConverter,
    'optimizer': OptimizerFromStringConverter,
    'policy': PolicyFromStringConverter,
    'learning_strategy_handler': LearningStrategyHandlerFromStringConverter,
    'testing_strategy_handler': TestingStrategyHandlerFromStringConverter,
    'label_annotator': LabelAnnotatorFromStringConverter
}

def __attempt_from_string_conversion(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        converter_type = CONVERTER_TYPE_MAP[key]
        name = value.get('name', None)
        params = value.get('parameters', None)
        for param_key, param_value in params.items():
            params[param_key] = __attempt_from_string_conversion(param_value, param_key)
        converter_instance = converter_type(**params)
        return converter_instance.convert_from_string(name)
    else:
        return value

def __get_local_path(file_path: str) -> str:
        try:
            url_parsed = urllib.parse.urlparse(file_path)
            if url_parsed.netloc != '' and url_parsed.scheme != '':
                if url_parsed.scheme == 's3':
                    logging.info('Loading file from S3 bucket...')
                    aws_handler = AWSHandler(os.getenv('ROLE_NAME'))
                    file_name = '/'.join(file_path.split('/')[3:])
                    aws_handler.download_file_from_s3(os.getenv('BUCKET_NAME'), file_name)
                else:
                    logging.info('Loading file from public URL...')
                    response = urllib.request.urlopen(file_path)
                    response.raise_for_status()
                    local_file = open(file_path.split('/')[-1], 'wb')
                    local_file.write(response.read())
                local_path = os.getcwd() + '/' + file_path.split('/')[-1]
            else:
                logging.info('Loading file from local path...')
                local_path = file_path
        except Exception as e:
            logging.error(f'Failed to get local path for {file_path}!')
            logging.error(e)
            raise e

        return local_path

def main(config_path: str, invoked_inside_gradient: bool = False) -> None:
    # try:
        config_local_path = __get_local_path(config_path)
        config = json.load(open(config_local_path, 'r'))
        for key, value in config['training_config'].items():
            config['training_config'][key] = __attempt_from_string_conversion(value, key)

        data_set_name = config['data_set_name']
        config['training_config']['data_path'] = __get_local_path(data_set_name)

        callbacks = []
        callback_dict = config.get('callbacks', None)
        if callback_dict is not None:
            for key, value in callback_dict.items():
                callbacks.append(CallbackFromStringConverter(**value).convert_from_string(key))

        weights_load_path = None
        weights_file_name = config.get('weights_file_name', None)
        if weights_file_name is not None:
            weights_load_path = __get_local_path(weights_file_name)

        training_handler = TrainingHandler(TrainingConfig(**config['training_config']))
        training_handler.run_training(callbacks = callbacks, weights_load_path = weights_load_path)

        report_name = f"Report from {datetime.now().__format__('%Y-%m-%d_%H_%M_%S')}.pdf"
        report_path = os.getcwd() + '\\' + report_name
        training_handler.generate_report(report_path)
        aws_handler = AWSHandler(os.getenv('ROLE_NAME'))
        aws_handler.upload_file_to_s3(os.getenv('BUCKET_NAME'), report_path, report_name)

    # except Exception as e:
    #     logging.error('Encounter problem during script execution!')
    #     logging.error(e)

        if invoked_inside_gradient:
            gradient_handler = GradientHandler()
            try:
                gradient_handler.delete_notebook(os.getenv('HOSTNAME'))
            except Exception as e:
                logging.error('Notebook was not deleted!')
                logging.error(e)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "{asctime} | {levelname} | {message}",
                        style="{", datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description = 'Runs training described by configuration file.')
    parser.add_argument('--config_path', type = str, required = True,
                        help = 'Path to configuration file in *json format.')
    parser.add_argument('--gradient', action = 'store_true', default = False,
                        help = 'Indicates if it was run on a gradient notebook that should be closed at the end.')

    args = parser.parse_args()
    main(args.config_path, args.gradient)