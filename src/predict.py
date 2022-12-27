import sys
sys.path.insert(0, './src')

from types import SimpleNamespace
from Helpers import *
from CLIConfig import load_inputs_defaults, set_torch_device, create_image_batch

def init_config():
  input_dict = load_inputs_defaults()
  config = SimpleNamespace(**input_dict)
  config = set_torch_device(config)
  return config

def predict(image_folder=None, image_src=None, device=0):
  config = init_config()
  config.use_gpu = device
  config = set_torch_device(config)
  if not image_folder and not image_src:
    raise Exception('Specify image folder or image path')
  elif image_folder: 
    maybe_terminate(path=image_folder, item_name='Images Folder')
    config.image_folder = image_folder
  elif image_src: 
    maybe_terminate(path=image_src, item_name='Image Path')
    config.image_src = image_src

  image_batch = create_image_batch(config)
  return image_batch.process(config)