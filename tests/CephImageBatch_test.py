import os
import pytest
import unittest
import numpy as np
from skimage import io

from src.pyceph.CephImageBatch import *

@pytest.fixture
def newCephImageBatch_single():
  test_image_src = 'tests/test-images/001.jpg'
  return CephImageBatch(img_path=test_image_src)

@pytest.fixture
def newCephImageBatch_multi():
  test_images_folder = 'tests/test-images'
  return CephImageBatch(img_folder=test_images_folder)

def test_single_batch_size(newCephImageBatch_single):
  obj = newCephImageBatch_single
  assert len(obj.batch) == 1

def test_single_batch_attr(newCephImageBatch_single):
  obj = newCephImageBatch_single
  ceph_image_obj = obj.batch[0]
  img_path = os.path.abspath('tests/test-images/001.jpg')
  assert img_path == ceph_image_obj.filename == img_path
  assert np.array_equal(ceph_image_obj.image, io.imread(img_path)) == True

def test_multi_batch_size(newCephImageBatch_multi):
  obj = newCephImageBatch_multi
  assert len(obj.batch) == 4
