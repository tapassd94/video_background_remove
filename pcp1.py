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
#video.subclip(0,50).ipython_display(width=300)
video.ipython_display(maxduration=200, width=300)
print(video.duration, video.size, video.fps)

# # transform video into a matrix
# def create_data_matrix_from_video(clip, k=5, scale=50):
#     return np.vstack([scipy.misc.imresize(rgb2gray(clip.get_frame(i/float(k))).astype(int),
#                       scale).flatten() for i in range(k * int(clip.duration))]).T

# def create_data_matrix_from_video(clip, k=5, dims=(60,80)):
#     return np.vstack([np.array(Image.fromarray(rgb2grey(video.get_frame(i/float(k))).astype(int)).resize(size=dims)).flatten() for i in range(k * int(clip.duration))]).T


# def create_data_matrix_from_video(clip, k, dims):
#     frames = []
#     for i in range(k * int(clip.duration)):
#         frame = clip.get_frame(i / float(k))
#         frame=cv2.resize(frame, dims)
#         frame = rgb2grey(frame).astype(int)
#         frame=frame.flatten()
#         frames.append(frame)
#     return np.vstack(frames).T # stack images horizontally

# def rgb2grey(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

from skimage.transform import resize

def rgb_grey(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def create_data_matrix_from_video(clip, k=5, dims=(80,60)):
    return np.vstack([resize(rgb_grey(clip.get_frame(i/float(k))).astype(int),
                      dims).flatten() for i in range(k * int(clip.duration))]).T

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
original_width = video.size[0] #320
original_height = video.size[1] #240
#dims = (int(240 * (scale/100)), int(320 * (scale/100)))
dims = (int(original_width * scale / 100), int(original_height * scale / 100))
print(dims) # single frame dimensions (height x width) 80X60

fps = 20
M = create_data_matrix_from_video(video, fps, dims)
np.save('surveillance_matrix.npy', M)
#M = np.load("surveillance_matrix.npy")
print(M.shape)

TOL=1e-7
def converged(Z, d_norm):
    err = np.linalg.norm(Z, 'fro') / d_norm
    print('error: ', err)
    return err < TOL
	
def shrink(M, tau):
    S = np.abs(M) - tau
    return np.sign(M) * np.where(S>0, S, 0)
	
def _svd(M, rank):
    return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)
	
def norm_op(M):
    return _svd(M, 1)[1][0]
	
def svd_reconstruct(M, rank, min_sv):
    u, s, v = _svd(M, rank)
    s -= min_sv
    nnz = (s > 0).sum()
    return u[:,:nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz
	
def pcp(X, maxiter=10, k=10): # refactored
    m, n = X.shape
    trans = m<n
    if trans: X = X.T; m, n = X.shape
        
    lamda = 1.0 / np.sqrt(np.max(M.shape))
    op_norm = norm_op(X)
    Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf) / lamda)
    mu = k*1.25/op_norm; mu_bar = mu * 1e7; rho = k * 1.5
    
    d_norm = np.linalg.norm(X, 'fro')
    L = np.zeros_like(X); sv = 1
    
    examples = []
    
    for i in range(maxiter):
        print("rank sv:", sv)
        X2 = X + Y/mu
        
        # update estimate of Sparse Matrix by "shrinking/truncating": original - low-rank
        S = shrink(X2 - L, lamda/mu)
        
        # update estimate of Low-rank Matrix by doing truncated SVD of rank sv & reconstructing.
        # count of singular values > 1/mu is returned as svp
        L, svp = svd_reconstruct(X2 - S, sv, 1/mu)
        
        # If svp < sv, you are already calculating enough singular values.
        # If not, add 20% (in this case 240) to sv
        sv = svp + (1 if svp < sv else round(0.05*n))
        
        # residual
        Z = X - L - S
        Y += mu*Z; mu *= rho
        
        examples.extend([S[140,:], L[140,:]])
        
        if m > mu_bar: m = mu_bar
        if converged(Z, d_norm): break
    
    if trans: L=L.T; S=S.T
    return L, S, examples
	
L, S, examples =  pcp(M, maxiter=5, k=10)

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