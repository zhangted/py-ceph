import unittest
from src.CLIConfig import *

class CLIConfigTest(unittest.TestCase):
  def test_create_empty_CLI_config(self):
    config = create_CLI_config(validate=False)
    assert config.image_folder is None
    assert config.image_src is None
    assert config.image_scale
    assert config.model_path
    assert config.use_gpu
    assert config.landmarksNum
    assert config.R1
    assert config.R2
