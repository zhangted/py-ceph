import os
import torch
from pyceph.ConsoleMsg import ConsoleMsg

def check_path(path):
  return os.path.exists(os.path.abspath(path))

def maybe_terminate(path=None, item_name=None, from_cli=True):
  if check_path(path): return
  abs_path = os.path.abspath(path)
  item_name = f"for {item_name}" if item_name else ''
  terminate(f"Invalid path {item_name}: {abs_path}", from_cli)

def terminate(msg, from_cli=False):
  if from_cli: ConsoleMsg.print_err_terminate(msg)
  else: raise Exception(msg)

def clean_path(path):
  return path.replace("'", '').replace('"', '') if path else None

def torch_device_str(arg):
  cuda_is_available = torch.cuda.is_available()
  if arg == 'cpu' or not cuda_is_available: return 'cpu'

  cuda_devices = torch.cuda.device_count()
  try:
    gpu = int(arg)
    if 0 <= gpu < cuda_devices: return f"cuda:{gpu}"
    raise Exception(f"Torch tried to select nonexistent cuda:{gpu}. Defaulting to cuda:0.")
  except Exception as e:
    print(e)
    return "cuda:0"