#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Aug 2023

@author: Guillaume Le Treut

Check TODO
"""
#==============================================================================
# libraries
#==============================================================================
from pathlib import Path
import logging
import numpy as np
import h5py
# import utils.cupy

logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

#-------------------------------------------------------------------
# functions
#-------------------------------------------------------------------
def potential(x,y):
  """
  return the Mueller-Brown potential
  """
  A_list = [-200., -100., -170., 15.]
  a_list = [-1.,   -1.,    -6.5, 0.7]
  b_list = [0.,    0.,     11.,  0.6]
  c_list = [-10.,  -10.,   -6.5, 0.7]

  x0_list = [1.,   0.,     -0.5, -1.]
  y0_list = [0.,   0.5,    1.5,  1.]

  return np.sum([A*np.exp( a*(x - x0)**2 + b*(x-x0)*(y-y0) + c*(y-y0)**2 ) for A,a,b,c,x0,y0 in zip(A_list, a_list, b_list, c_list, x0_list, y0_list)], axis=0)

def potential_grad(x,y):
  """
  return the gradient of the Mueller-Brown potential
  """
  A_list = [-200., -100., -170., 15.]
  a_list = [-1.,   -1.,    -6.5, 0.7]
  b_list = [0.,    0.,     11.,  0.6]
  c_list = [-10.,  -10.,   -6.5, 0.7]

  x0_list = [1.,   0.,     -0.5, -1.]
  y0_list = [0.,   0.5,    1.5,  1.]

  fis = np.array([
    A*np.exp( a*(x - x0)**2 + b*(x-x0)*(y-y0) + c*(y-y0)**2 )
    for A,a,b,c,x0,y0 in zip(A_list, a_list, b_list, c_list, x0_list, y0_list)
    ])

  grad = np.array([
    [2*a*(x-x0) + b*(y-y0), 2*c*(y-y0) + b*(x-x0)]
    for a,b,c,x0,y0 in zip(a_list, b_list, c_list, x0_list, y0_list)
    ])

  grad = np.einsum('nd,n->d', grad, fis)
  return grad

#==============================================================================
# processing
#==============================================================================
def plot_potential(XYs, Us, extent, x_minima=None, levels=2**5, title='Mueller-Brown potential', lw=0.5, ms=2, ax=None):
  """
  Base plot for the Mueller-Brown potential
  """
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
  import matplotlib.colors as mco
  import scipy.ndimage

  N = len(XYs)
  sN = int(np.sqrt(N))
  xmesh, ymesh = np.einsum('xyd->dxy', XYs.reshape(sN,sN,-1))
  umesh = Us.reshape(sN,sN)

  if ax is None:
    fig = plt.figure()
    ax = fig.gca()
  else:
    fig = ax.get_figure()

  Z = umesh.T
  Z = scipy.ndimage.zoom(Z, 3)
  ax.plot(XYs[:,0], XYs[:,1], ls='none', marker='+', ms=ms, mfc='white', mec='k', mew=lw)

  cs = ax.contour(Z, levels=levels, origin='lower', extend='both',
                  linewidths=lw, extent=extent, colors='k')

  ax.set_title(title, fontsize='medium')

  # We can still add a colorbar for the image, too.
  # cbi = fig.colorbar(im, orientation='vertical', shrink=1.0)
  ax.set_aspect('equal')
  ax.tick_params(right=True, top=True, length=4)

  # mark minima
  if not (x_minima is None):
    ax.plot(x_minima[:,0], x_minima[:,1], 'ro', ms=ms)

  fig.tight_layout()

  return fig

def plot_trajectories(outputdir, Ts, XYs, xlim, ylim, XYs_static=None, taillength=5, lw=0.5, ms=2, figsize=(4,3), dpi=300, aspect=None, ntraj_max=None, func_background=None, fprefix="state_"):
  """
  Plot the mean trajectory
  """
  import matplotlib.pyplot as plt
  from matplotlib.patches import Polygon
  from scipy.spatial import ConvexHull

  ntraj, npts, dim = XYs.shape
  if len(Ts) != npts:
    raise ValueError("Ts and XYs must have the same size")

  if dim != 2:
    raise ValueError("Plot only works for dimension 2!")

  if ntraj_max is None:
    ntraj_max = ntraj

  tfmt = "t = {:0" + "{:d}".format(int(np.ceil(np.log10(Ts[-1])))) + ".{:d}".format(-int(np.floor(np.log10(Ts[1])))) + "f}"
  ffmt = fprefix + "{:0" + "{:d}".format(int(np.ceil(np.log10(npts)))) + "d}"
  for k in range(npts):
  # for k in range(10):
    if func_background is None:
      fig = plt.figure(facecolor='w', figsize=figsize)
      ax = fig.gca()
      if not (XYs_static is None):
        ax.plot(XYs_static[:,0], XYs_static[:,1], ls='none', marker='o', ms=ms, mfc='white', mec='k', mew=lw)
      ax.set_xlim(*xlim)
      ax.set_ylim(*ylim)
      ax.set_aspect('equal')
      ax.set_axis_off()

    else:
      fig = func_background()
      ax = fig.gca()

    # compute the mean tail
    x_tails, y_tails = np.einsum('ntd->dnt', XYs[:, max(0, k-taillength):k+1])[:, :ntraj_max]
    for x,y in zip(x_tails, y_tails):
      lines=ax.plot(x, y, '-', lw=lw)
      linecolor=lines[0].get_color()
      ax.plot([x[-1]], [y[-1]], 'o', ms=ms,color=linecolor)

    ax.set_title(tfmt.format(Ts[k]), fontsize='medium')

    fpath = outputdir / (ffmt.format(k) + ".png")
    fig.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=dpi)
    logging.debug("File written: {:s}".format(str(fpath)))
    fig.clf()
    plt.close('all')

  return

def make_movie(fpath, file_list, fps=30):
  """
  Make a movie from the input file list
  """
  import imageio

  imageio.mimsave(fpath, [imageio.imread(f) for f in file_list], fps=fps)
  print("Written file {:s}".format(str(fpath)))

  return
