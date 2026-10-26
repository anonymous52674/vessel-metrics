# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lookup tables used by surface distance metrics."""

import math

import numpy as np


# In the neighbourhood, points are ordered: top left, top right, bottom left,
# bottom right.
ENCODE_NEIGHBOURHOOD_2D_KERNEL = np.array([[8, 4], [2, 1]])


def create_table_neighbour_code_to_contour_length(spacing_mm):
  """Returns an array mapping neighbourhood code to the contour length.

  For the list of possible cases and their figures, see page 38 from:
  https://nccastaff.bournemouth.ac.uk/jmacey/MastersProjects/MSc14/06/thesis.pdf

  In 2D, each point has 4 neighbors. Thus, are 16 configurations. A
  configuration is encoded with '1' meaning "inside the object" and '0' "outside
  the object". The points are ordered: top left, top right, bottom left, bottom
  right.

  The x0 axis is assumed vertical downward, and the x1 axis is horizontal to the
  right:
   (0, 0) --> (0, 1)
     |
   (1, 0)

  Args:
    spacing_mm: 2-element list-like structure. Voxel spacing in x0 and x1
      directions.
  """
  neighbour_code_to_contour_length = np.zeros([16])

  vertical = spacing_mm[0]
  horizontal = spacing_mm[1]
  diag = 0.5 * math.sqrt(spacing_mm[0]**2 + spacing_mm[1]**2)
  # pyformat: disable
  neighbour_code_to_contour_length[int("00"
                                       "01", 2)] = diag

  neighbour_code_to_contour_length[int("00"
                                       "10", 2)] = diag

  neighbour_code_to_contour_length[int("00"
                                       "11", 2)] = horizontal

  neighbour_code_to_contour_length[int("01"
                                       "00", 2)] = diag

  neighbour_code_to_contour_length[int("01"
                                       "01", 2)] = vertical

  neighbour_code_to_contour_length[int("01"
                                       "10", 2)] = 2*diag

  neighbour_code_to_contour_length[int("01"
                                       "11", 2)] = diag

  neighbour_code_to_contour_length[int("10"
                                       "00", 2)] = diag

  neighbour_code_to_contour_length[int("10"
                                       "01", 2)] = 2*diag

  neighbour_code_to_contour_length[int("10"
                                       "10", 2)] = vertical

  neighbour_code_to_contour_length[int("10"
                                       "11", 2)] = diag

  neighbour_code_to_contour_length[int("11"
                                       "00", 2)] = horizontal

  neighbour_code_to_contour_length[int("11"
                                       "01", 2)] = diag

  neighbour_code_to_contour_length[int("11"
                                       "10", 2)] = diag
  # pyformat: enable

  return neighbour_code_to_contour_length
