import unittest
from unittest import mock
import pytest
import torch

from src.CLIConfig import *
from src.ModelWrapper import *

# https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212
def validate_state_dicts(model_state_dict_1, model_state_dict_2):
  if len(model_state_dict_1) != len(model_state_dict_2):
      return False

  # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
  if next(iter(model_state_dict_1.keys())).startswith("module"):
      model_state_dict_1 = {
          k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
      }

  if next(iter(model_state_dict_2.keys())).startswith("module"):
      model_state_dict_2 = {
          k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
      }

  for ((k_1, v_1), (k_2, v_2)) in zip(
      model_state_dict_1.items(), model_state_dict_2.items()
  ):
      if k_1 != k_2:
          return False
      # convert both to the same CUDA device
      if str(v_1.device) != "cuda:0":
          v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
      if str(v_2.device) != "cuda:0":
          v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

      if not torch.allclose(v_1, v_2):
          return False

  return True

@pytest.fixture
def get_config_torch_cpu():
  with mock.patch('torch.cuda.is_available', lambda: True):
    return create_CLI_config(validate=False)

@pytest.fixture
def newModelWrapper(get_config_torch_cpu):
  return ModelWrapper(get_config_torch_cpu)

def test_load_model(newModelWrapper):
  modelWrapper = newModelWrapper
  model_path = modelWrapper.model_path
  model1 = modelWrapper.load_model()

  with gzip.open(model_path, 'rb') as zipped_model:
    model2 = torch.load(zipped_model, map_location=torch.device('cpu'))
    assert validate_state_dicts(model1.state_dict(), model2.state_dict())
