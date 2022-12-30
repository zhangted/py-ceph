import argparse
import yaml
import pkg_resources

from pyceph.Helpers import torch_device_str, clean_path, terminate, maybe_terminate
from pyceph.CephImageBatch import CephImageBatch

def clean_paths(config):
  config.image_folder = clean_path(config.image_folder)
  config.image_src = clean_path(config.image_src)
  return config

def set_torch_device(config):
  config.use_gpu = torch_device_str(config.use_gpu)
  return config

def load_inputs_defaults():
  try:
    input_yaml_path = pkg_resources.resource_filename(__name__, 'input.yml')
    print(f"loading inputs from: {input_yaml_path}")
    with open(input_yaml_path, 'r') as stream:
      return yaml.safe_load(stream)
  except Exception as e: terminate(e)

def create_CLI_config(validate=True):
  input_dict = load_inputs_defaults()
  parser = argparse.ArgumentParser()

  for arg_name in input_dict.keys():
    parser.add_argument(f"--{arg_name}", default=input_dict[arg_name])

  config = parser.parse_args()
  config = set_torch_device(config)

  if not validate: return config
  return validate_input(clean_paths(config))

def validate_input(config):
  if(config.image_folder and config.image_src):
    terminate('Cannot specify both batch folder AND single image src to be processed', from_cli=True)
  elif(config.image_folder is None and config.image_src is None):
    terminate('Must specify batch folder OR single image to be processed', from_cli=True)
  elif(config.image_folder):
    maybe_terminate(path=config.image_folder, item_name='Image Folder')
    return config
  elif(config.image_src):
    maybe_terminate(path=config.image_src, item_name='Image Src')
    return config

def create_image_batch(config):
  if config.image_src:
    return CephImageBatch(img_path=config.image_src)
  elif config.image_folder:
    return CephImageBatch(img_folder=config.image_folder)