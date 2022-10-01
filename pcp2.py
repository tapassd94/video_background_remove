import os
import sys
import scipy
import numpy as np
import cv2
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
from scipy import stats
from matplotlib import pyplot as plt, rcParams, rc
from sklearn import decomposition

from scipy import sparse
from sklearn.utils.extmath import randomized_svd
import fbpca

%matplotlib inline
%precision 4
%config InlineBackend.figure_format = 'retina'

rc('animation', html='html5')
rcParams['figure.figsize'] = 8, 10
np.set_printoptions(precision=4, linewidth=100)

from __future__ import division, print_function

__all__ = ["pcp"]

import time
import fbpca
import logging
import numpy as np
from scipy.sparse.linalg import svds

# load the videos
video = mpy.VideoFileClip('C:/Users/Dell/Desktop/IIMB/data/video/Video_003.avi')
video.ipython_display(width=300)
print(video.duration, video.size, video.fps)

# # transform video into a matrix
# def create_data_matrix_from_video(clip, k, scale):
#     frames = []
#     for i in range(k * int(clip.duration)):
#         frame = clip.get_frame(i / float(k))
#         frame = rgb2grey(frame).astype(int)
#         frame = scipy.misc.imresize(frame, scale).flatten()
#         frames.append(frame)
#     return np.vstack(frames).T # stack images horizontally

def create_data_matrix_from_video(clip, k, dims):
    frames = []
    for i in range(k * int(clip.duration)):
        frame = clip.get_frame(i / float(k))
        frame=cv2.resize(frame, dims)
        frame = rgb2grey(frame).astype(int)
        frame=frame.flatten()
        frames.append(frame)
    return np.vstack(frames).T # stack images horizontally

def rgb2grey(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def plot_images(M, A, E, index_array, dims, filename=None):
    f = plt.figure(figsize=(15, 10))
    r = len(index_array)
    pics = r * 3
    for k, i in enumerate(index_array):
        for j, mat in enumerate([M, A, E]):
            sp = f.add_subplot(r, 3, 3*k + j + 1)
            sp.axis('Off')
            if isinstance(pixels, scipy.sparse.csr_matrix):
                pixels = pixels.todense()
            plt.imshow(np.reshape(pixels, dims), cmap='gray')
    return f

# change resolution of image
scale =25    # scale to X percent (100 means no scaling). CHANGE THIS FOR BETTER RESOLUTION
original_width = video.size[1]
original_height = video.size[0]

dims = (int(original_width * scale / 100), int(original_height * scale / 100))
print(dims) # single frame dimensions (height x width)

fps = 20
M = create_data_matrix_from_video(video, fps, dims)
np.save('video_matrix', M)
print(M.shape)

def shrink(M, tau):
    sgn = np.sign(M)
    S = np.abs(M) - tau
    S[S < 0.0] = 0.0
    return sgn * S

def converged(err, delta):
    return err < delta

def _svd(method, X, rank, tol, **args):
    rank = min(rank, np.min(X.shape))
    if method == "approximate":
        return fbpca.pca(X, k=rank, raw=True, **args)
    elif method == "exact":
        return np.linalg.svd(X, full_matrices=False, **args)
    elif method == "sparse":
        if rank >= np.min(X.shape):
            return np.linalg.svd(X, full_matrices=False)
        u, s, v = svds(X, k=rank, tol=tol)
        u, s, v = u[:, ::-1], s[::-1], v[::-1, :]
        return u, s, v
    raise ValueError("invalid SVD method")
	
def pcp(M, delta=1e-7, mu=None, maxiter=500, verbose=False, missing_data=True,
        svd_method="approximate", **svd_args):
    # Check the SVD method.
    allowed_methods = ["approximate", "exact", "sparse"]
    if svd_method not in allowed_methods:
        raise ValueError("'svd_method' must be one of: {0}"
                         .format(allowed_methods))

    # Check for missing data.
    shape = M.shape
    if missing_data:
        missing = ~(np.isfinite(M))
        if np.any(missing):
            M = np.array(M)
            M[missing] = 0.0
    else:
        missing = np.zeros_like(M, dtype=bool)
        if not np.all(np.isfinite(M)):
            logging.warn("The matrix has non-finite entries. "
                         "SVD will probably fail.")

    # Initialize the tuning parameters.
    lam = 1.0 / np.sqrt(np.max(shape))
    if mu is None:
        mu = 0.25 * np.prod(shape) / np.sum(np.abs(M))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = np.sum(M ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)
    S = np.zeros(shape)
    Y = np.zeros(shape)
    d_norm = np.linalg.norm(M, 'fro')
    while i < max(maxiter, 1):
        # SVD step.
        strt = time.time()
        u, s, v = _svd(svd_method, M - S + Y / mu, rank+1, 1./mu, **svd_args)
        svd_time = time.time() - strt

        s = shrink(s, 1./mu)
        rank = np.sum(s > 0.0)
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        L = np.dot(u, np.dot(np.diag(s), v))

        # Shrinkage step.
        S = shrink(M - L + Y / mu, lam / mu)

        # Lagrange step.
        step = M - L - S
        step[missing] = 0.0
        Y += mu * step

        # Check for convergence.
        err = np.linalg.norm(step, 'fro') / d_norm
        if verbose:
            print(("Iteration {0}: error={1:.5e}, rank={2:d}, nnz={3:d}, "
                   "time={4:.3e}")
                  .format(i, err, np.sum(s > 0), np.sum(S > 0), svd_time))
        if converged(err, delta):
            break
        i += 1

    if i >= maxiter:
        logging.warn("convergence not reached in pcp")
    return L, S, (u, s, v)


	
L, S, (u, s, v) = pcp(M, maxiter=100, verbose=True, svd_method="approximate")

f = plt_images(M, S, L, [0, 100, 1000], dims)

people_frames = S.reshape((dims[0], dims[1], -1))

fig, ax = plt.subplots()
def make_frame(t):
    ax.clear()
    ax.imshow(people_frames[...,int(t*fps)])
    return mplfig_to_npimage(fig)

animation = mpy.VideoClip(make_frame, duration=int(video.duration-1))
animation.write_videofile('people1.mp4', fps=fps)

AA1 = mpy.VideoFileClip('people1.mp4')
AA1.ipython_display(maxduration=200, width=300)