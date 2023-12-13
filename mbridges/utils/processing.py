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
def register_trajectories(traj_list, tmin=0, tmax=None, dt=1.):
  '''
  Register trajectories so that we obtain trajectories of coordinates at a regular time subdivision.
  INPUT:
    * traj_list [N,d]: a list of N trajectories
    * tmin [float]: lower end of the time window
    * tmax [float]: upper end of the time window
    * dt [float]: time interval of the subdivision
  '''
  # extract trajectories minimum and maximum times
  Ts = []
  for k,traj in enumerate(traj_list):
    T = traj[[0,-1],0]
    Ts.append(T)
  Ts = np.concatenate(Ts).astype('float64')
  Tsmin = np.min(Ts)
  Tsmax = np.max(Ts)

  # define the registered times
  ## define bounds
  if tmin is None:
    tmin = Tsmin
  if tmax is None:
    tmax = Tsmax
  if tmin > Tsmin:
    logging.info("tmin = {:g} > min(T) = {:g}".format(tmin, Tsmin))
  if tmax < Tsmax:
    logging.info("tmax = {:g} < max(T) = {:g}".format(tmax, Tsmax))

  ## define registered times
  nbins = max(1, int((tmax-tmin) / dt))
  dt = (tmax-tmin) / nbins # redefine dt to correspond to an integer number of bins
  t_regis = tmin + np.arange(nbins+1, dtype='float64') * dt # regular time subdivision
  logging.info("dt = {:g}  nbins = {:d}  tmin = {:g}  tmax = {:g}".format(dt, nbins, tmin, tmax))

  # create the registered trajectories
  traj_regis_list = []
  for traj in traj_list:
    T = traj[:, 0]
    states = traj[:, 1]

    idx = np.digitize(t_regis, T)-1 # place registered times in bins such that t_{i-1} <= t_regis < t_{i} is associated with state i-1.
    idx[t_regis < T[0]] = 0 # registered times strictly smaller than smallest trajectory time are set in the initial state at i=0.
    idx[~(t_regis < T[-1])] = len(T)-1 # registered times larger than the largest trajectory time are left in the final state at i = len(T) - 1

    states_regis = states[idx] # get the values of the assigned states

    traj_regis_list.append(states_regis)
  return t_regis, np.array(traj_regis_list)

