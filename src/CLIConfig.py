import argparse
from Helpers import torch_device_str, clean_path, check_path, terminate, maybe_terminate
from CephImageBatch import CephImageBatch

def clean_paths(config):
  config.image_folder = clean_path(config.image_folder)
  config.image_src = clean_path(config.image_src)
  return config

def create_CLI_config():
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_folder", default=None, type=str)
  parser.add_argument("--image_src", default=None, type=str)
  parser.add_argument("--image_scale", default=(800, 640), type=tuple)
  parser.add_argument("--model_path", default='pretrained_models/12-26-22.pkl.gz', type=str)
  parser.add_argument("--use_gpu", default=torch_device_str(0), type=torch_device_str)
  config = parser.parse_args()
  config.landmarksNum = 19
  config.R1, config.R2 = 41, 41
  return validate_input(clean_paths(config))

def validate_input(config):
  if(config.image_folder and config.image_src):
    terminate('Cannot specify both batch folder AND single image src to be processed')
  elif(config.image_folder is None and config.image_src is None):
    terminate('Must specify batch folder OR single image to be processed')
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