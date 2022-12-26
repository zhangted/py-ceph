import torch
import argparse
import pkg_resources
import os
import gzip
import sys
sys.path.insert(0, './src')

import models
from CephImageBatch import CephImageBatch
from ConsoleMsg import ConsoleMsg

RELATIVE_MODEL_PATH = 'pretrained_models/12-24-22.zip'

def check_path(path, item_name=None):
  abs_path = os.path.abspath(path)
  if not os.path.exists(abs_path):
    item_name = f"for {item_name}" if item_name else ''
    ConsoleMsg.print_err_terminate(f"Invalid path {item_name}: {abs_path}")
  return True

def clean_paths(config):
  def clean(path):
    return path.replace("'", '').replace('"', '') if path else None
  config.image_folder = clean(config.image_folder)
  config.image_src = clean(config.image_src)
  return config

def use_cuda():
  if torch.cuda.is_available(): return "cuda:0"
  return 0

def get_image_batch():
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_folder", default=None, type=str)
  parser.add_argument("--image_src", default=None, type=str)
  parser.add_argument("--image_scale", default=(800, 640), type=tuple)
  parser.add_argument("--use_gpu", type=int, default=use_cuda())
  parser.add_argument("--R1", type=int, default=41)
  parser.add_argument("--R2", type=int, default=41)
  config = parser.parse_args()
  config.landmarksNum = 19
  config = clean_paths(config)

  # input validation
  if(config.image_folder and config.image_src):
    ConsoleMsg.print_err_terminate('Cannot specify both batch folder AND single image src to be processed')
  elif(config.image_folder is None and config.image_src is None):
    ConsoleMsg.print_err_terminate('Must specify batch folder OR single image to be processed')
  elif(config.image_folder):
    check_path(config.image_folder, 'Image Folder')
    return config, CephImageBatch(img_folder=config.image_folder)
  elif(config.image_src):
    check_path(config.image_src, 'Image Src')
    return config, CephImageBatch(img_path=config.image_src)

def main():
  config, image_batch = get_image_batch()

  model_path = pkg_resources.resource_filename(__name__, RELATIVE_MODEL_PATH)
  check_path(model_path, "Pretrained Model")


  kwargs = { } if config.use_gpu else { 'map_location': torch.device('cpu') }
  with gzip.open(model_path, 'rb') as zipped_model:
    model = torch.load(zipped_model, **kwargs)

  image_batch.process(model, config)

if __name__ == '__main__':
  main()




'''
# save model as gzip
import gzip
import torch

# Save the model
torch.save(model.state_dict(), 'model.pkl')

# Compress the model
with open('model.pkl', 'rb') as f_in, gzip.open('model.pkl.gz', 'wb') as f_out:
    f_out.writelines(f_in)


# load compressed model
import gzip
import torch

# Load the compressed model
with gzip.open('model.pkl.gz', 'rb') as f:
    model = torch.load(f)
'''