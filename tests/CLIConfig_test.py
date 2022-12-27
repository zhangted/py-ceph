import unittest
import pytest
from src.CLIConfig import *

@pytest.fixture
def get_default_config():
  return create_CLI_config(validate=False)

def test_create_empty_CLI_config(get_default_config):
  config = get_default_config
  assert config.image_folder is None
  assert config.image_src is None
  assert config.image_scale
  assert config.model_path
  assert config.use_gpu
  assert config.landmarksNum
  assert config.R1
  assert config.R2

def test_clean_paths(get_default_config):
  config = get_default_config
  config.image_src = '"hello"'
  config.image_folder = "'hello\"'"
  cleaned_config = clean_paths(config)
  expected = 'hello'
  assert cleaned_config.image_src == expected
  assert cleaned_config.image_folder == expected

