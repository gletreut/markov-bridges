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
logging.basicConfig(format='%(asctime)s [%(filename)s:%(lineno)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
import numpy as np

def plot_umap(XYs, labels, categoriesdict, lw=0.5, ms=2, figsize=None, ax=None, legend=True):
  """
  Base plot for the UMAP of the pancreas scRNA dataset
  """
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm
  import matplotlib.colors as mco
  import scipy.ndimage

  if ax is None:
    fig = plt.figure(facecolor='none', figsize=figsize)
    ax = fig.gca()
  else:
    fig = ax.get_figure()

  N = len(XYs)

  for cat in categoriesdict.keys():
    indices = np.arange(N)[labels == categoriesdict[cat]]
    ax.plot(XYs[indices,0], XYs[indices,1], 'o', ms=ms, label=cat)

  if legend:
    ax.legend(loc='upper left', bbox_to_anchor=(1.,1.))
  ax.set_aspect('equal')
  # ax.tick_params(right=False, top=False, length=4)
  ax.tick_params(right=False, top=False, left=False, bottom=False,
                 labelright=False, labeltop=False, labelleft=False, labelbottom=False,
                 length=4)

  for el in 'left', 'bottom', 'right', 'top':
    ax.spines[el].set_visible(False)

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
