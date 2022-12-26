import unittest
from unittest import mock

from src.Helpers import *

class HelpersTest(unittest.TestCase):

  def test_check_invalid_path(self):
    return self.assertFalse(check_path('nonexistent_folder'))

  def test_check_valid_path(self):
    return self.assertTrue(check_path('tests/test-images'))

  def test_clean_path_none_input(self):
    path = None
    return self.assertTrue(clean_path(path) == None)

  def test_clean_path_quotes_apo_input(self):
    path = '"test_path\'"'
    return self.assertTrue(clean_path(path) == 'test_path')

  def test_torch_device_selection_cpu(self):
    return self.assertTrue(torch_device_str('cpu') == 'cpu')

  def test_torch_device_selection_no_cuda_device(self):
    with mock.patch('torch.cuda.is_available', lambda: False):
      return self.assertTrue(torch_device_str(0) == 'cpu')

  def test_torch_device_selection_negative(self):
    with mock.patch('torch.cuda.is_available', lambda: True):
      with mock.patch('torch.cuda.device_count', lambda: 2):
        return self.assertTrue(torch_device_str(-1) == 'cuda:0')

  def test_torch_device_selection_invalid_idx(self):
    with mock.patch('torch.cuda.is_available', lambda: True):
      with mock.patch('torch.cuda.device_count', lambda: 2):
        return self.assertTrue(torch_device_str(3) == 'cuda:0')

  def test_torch_device_selection_valid_idx(self):
    with mock.patch('torch.cuda.is_available', lambda: True):
      with mock.patch('torch.cuda.device_count', lambda: 2):
        return self.assertTrue(torch_device_str(1) == 'cuda:1')