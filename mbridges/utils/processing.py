#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on 2023-08-16

@author: Guillaume Le Treut - guillaume.letreut@czbiohub.org
'''

#==============================================================================
# libraries
#==============================================================================
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


#==============================================================================
# functions
#==============================================================================
def register_times(times, t0=0, tN=None, dt=1.):
  '''
  Register trajectories so that we obtain trajectories of coordinates at a regular time subdivision.
  INPUT:
    * times [N,]: a list of N time series
    * tmin [float]: lower end of the time window
    * tmax [float]: upper end of the time window
    * dt [float]: time interval of the subdivision
  '''
  Nt = len(times)
  # extract trajectories minimum and maximum times
  tmin = np.min(times)
  tmax = np.max(times)

  # define the registered times
  ## define bounds
  if t0 is None:
    t0 = tmin
  if tN is None:
    tN = np.ceil(tmax/dt)*dt # to ensure an integer number of bins
  if t0 > tmin:
    logging.info("t0 = {:g} > min(T) = {:g}".format(t0, tmin))
  if tN < tmax:
    logging.info("tN = {:g} < max(T) = {:g}".format(tN, tmax))

  ## define registered times
  nbins = max(1, int(np.ceil((tN-t0) / dt)))
  tN_ = t0 + nbins*dt
  if tN_ != tN:
    logging.info("Adjusting tN from {:g} to {:g}".format(tN, tN_))
    tN = tN_
  # dt = (tmax-tmin) / nbins # redefine dt to correspond to an integer number of bins
  t_regis = t0 + np.arange(nbins+1, dtype='float64') * dt # regular time subdivision
  logging.info("dt = {:g}  nbins = {:d}  t0 = {:g}  tN = {:g}".format(dt, nbins, t_regis[0], t_regis[-1]))

  # create the registered trajectories
  idx = np.digitize(t_regis, times)-1 # place registered times in bins such that t_{i-1} <= t_regis < t_{i} is associated with state i-1.
  idx[t_regis < tmin] = 0 # registered times strictly smaller than smallest trajectory time are set in the initial state at i=0.
  idx[~(t_regis < tmax)] = Nt-1 # registered times larger than the largest trajectory time are left in the final state at i = Nt - 1

  return t_regis, idx

