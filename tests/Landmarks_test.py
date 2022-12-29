import unittest
from src.pyceph.Landmarks import *

class LandmarksTest(unittest.TestCase):
  def test_num_landmarks(self):
    num_landmarks = len([landmark.value for landmark in Landmarks])
    return self.assertTrue(num_landmarks==19)