# scripts/train_model.py
 
import logging
import json

from source.training import TrainingHandler, TrainingConfig
from source.gradient import GradientHandler
from source.model import VGGceptionCnnBluePrint

# import argparse
# from datetime import datetime
# from typing import Optional

def str_to_model_blue_print(model_blue_print_str):
    blue_print_map = {
        'vggception': VGGceptionCnnBluePrint()
    }

    return blue_print_map.get(model_blue_print_str)

def main(config_path: str, invoked_inside_gradient: bool = False) -> None:

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

        training_config = TrainingConfig(**config['training_config']['reinterpretable'],
                                         model_blue_print = ,
                                         validator = 
                                         policy = ,
                                         optimizer = )
        
        training_handler = TrainingHandler()
        training_handler.run_training(callbacks = , weights_load_path = config['weights_load_path'])

        if invoked_inside_gradient:
            gradient_handler = 



    

    # try:
    #     gradient_handler = GradientHandler()
    #     notebook_id = gradient_handler.creata_notebook(url, command, notebook_name, machines,
    #                                                    timeout, str_to_dict(environment))
    #     if notebook_id is not None:
    #         logging.info(f'Instantiated training run successfully on notebook {notebook_id}.')

    # except Exception as e:
    #     logging.error('Encounter problem during script execution!')
    #     logging.error(e)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "{asctime} | {levelname} | {message}",
                        style="{", datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description = 'Instantiate training on Paperspace Gradient machine.')
    parser.add_argument('--url', type = str, required = True, 
                        help = 'Url to github repository that should be copied into machine.')
    parser.add_argument('--command', type = str, required = True, help = 'Command invoked at machine.')
    parser.add_argument('--name', type = str, required = False, help = 'Name of the notebook.')
    parser.add_argument('--machines', type = str, required = False,
                        help = '''List of machines, that looks like: machine_type_1,machine_type_2,...,machine_type_N.
                        Possible machine types are: e.g. Free-P5000, Free-A4000, Free-RTX4000, Free-RTX5000.''')
    parser.add_argument('--timeout', type = int, required = False, 
                        help = 'Timeout that machine will be terminated after. Maximal value is 6.')
    parser.add_argument('--env', type = str, required = False, 
                        help = 'String denoted dictionary of environmental variables, eg. key:value,key_2:value_2.')

    args = parser.parse_args()
    main(args.url, args.command, args.name, args.machines, args.timeout, args.env)