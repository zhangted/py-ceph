from enum import Enum

class Landmarks(Enum):
  '''Maps the model prediction array to respective cephalometric landmark names.'''

  SELLA = 0
  NASION = 1
  ORBITALE = 2
  PORION = 3
  SUBSPINALE = 4
  SUPRAMENTALE = 5
  POGONION = 6
  MENTON = 7
  GNATHION = 8
  GONION = 9
  INCISION_INFERIUS = 10
  INCISION_SUPERIUS = 11
  UPPER_LIP = 12
  LOWER_LIP = 13
  SUBNASALE = 14
  SOFT_TISSUE_POGONION = 15
  POSTERIOR_NASAL_SPINE = 16
  ANTERIOR_NASAL_SPINE = 17
  ARTICULARE = 18

  def __str__(self):
    return self.name.replace('_', ' ').lower().title()