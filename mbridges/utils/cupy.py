#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Dec 2022

@author: Guillaume Le Treut

"""

#==============================================================================
# libraries
#==============================================================================
import numpy as np
try:
  import cupy
except ImportError:
  cupy = None

#==============================================================================
# functions
#==============================================================================
def get_array_module(a):
  """
  Return the module of an array a
  """
  if cupy:
    return cupy.get_array_module(a)
  else:
    return np

