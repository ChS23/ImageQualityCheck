import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def ssim(
        img1: np.ndarray,
        img2: np.ndarray,
        K: tuple = None,
        window=None,
        L: float = None
) -> tuple:
    """% ========================================================================
% Edited code by Adam Turcotte and Nicolas Robidoux
% Laurentian University
% Sudbury, ON, Canada
% Last Modified: 2011-01-22
% ----------------------------------------------------------------------
% This code implements a refactored computation of SSIM that requires
% one fewer blur (4 instead of 5), the same number of pixel-by-pixel
% binary operations (10), and two fewer unary operations (6 instead of 8).
%
% In addition, this version reduces memory usage with in-place functions.
% As a result, it supports larger input images.
%========================================================================

% ========================================================================
% SSIM Index with automatic downsampling, Version 1.0
% Copyright(c) 2009 Zhou Wang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the
% Structural SIMilarity (SSIM) index between two images
%
% Please refer to the following paper and the website with suggested usage
%
% Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
% quality assessment: From error visibility to structural similarity,"
% IEEE Transactios on Image Processing, vol. 13, no. 4, pp. 600-612,
% Apr. 2004.
%
% http://www.ece.uwaterloo.ca/~z70wang/research/ssim/
%
% Note: This program is different from ssim_index.m, where no automatic
% downsampling is performed. (downsampling was done in the above paper
% and was described as suggested usage in the above website.)
%
% Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size
%            depends on the window size and the downsampling factor.
%
%Basic Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim, ssim_map] = ssim(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim, ssim_map] = ssim(img1, img2, K, window, L);
%
%Visualize the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%========================================================================
"""

    if img1.dtype != img2.dtype:
        raise ValueError("Input images must have the same dtype.")

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Input images must be 2D.")

    M, N = img1.shape

    if K is None and L is None and window is None:
        if (M < 11) or (N < 11):
            return -np.inf, -np.inf

        window = gaussian_filter(np.ones((11, 11)), 1.5)
        K = (0.01, 0.03)
        L = 255

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    f = max(1, round(min(M, N) / 256))

    if f > 1:
        lpf = np.ones((f, f)) / f ** 2
        img1 = convolve(img1, lpf, mode='reflect')
        img2 = convolve(img2, lpf, mode='reflect')

        img1 = img1[::f, ::f]
        img2 = img2[::f, ::f]

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    window = window / np.sum(np.sum(window))
    ssim_map = convolve(img1, window, mode='reflect')
    w1 = convolve(img2, window, mode='reflect')
    w2 = ssim_map * w1
    w2 = 2 * w2 + C1
    w1 = (w1 - ssim_map) ** 2 + w2
    ssim_map = convolve(img1 * img2, window, mode='reflect')
    ssim_map = (2 * ssim_map + (C1 + C2)) - w2
    ssim_map *= w2
    img1 = img1 ** 2
    img2 = img2 ** 2
    img1 = img1 + img2

    if C1 > 0 and C2 > 0:
        w2 = convolve(img1, window, mode='reflect')
        w2 = w2 - w1 + (C1 + C2)
        w2 *= w1
        ssim_map /= w2
    else:
        w3 = convolve(img1, window, mode='reflect')
        w3 = w3 - w1 + (C1 + C2)
        w4 = np.ones_like(w1)
        index = (w1 * w3 > 0)
        w4[index] = ssim_map[index] / (w1[index] * w3[index])
        index = (w1 != 0) & (w3 == 0)
        w4[index] = w2[index] / w1[index]
        ssim_map = w4

    return np.mean(ssim_map), ssim_map


if __name__ == '__main__':
    import sewar

    image1_path = r'C:\Users\c4s23\YandexDisk\GitHub\ImageQualityCheck\data\TID2013\distorted_images\i01_22_5.bmp'
    image2_path = r'C:\Users\c4s23\YandexDisk\GitHub\ImageQualityCheck\data\TID2013\reference_images\I01.BMP'

    from PIL import Image

    image1 = Image.open(image1_path).convert('L')
    image2 = Image.open(image2_path).convert('L')

    images1 = np.array(image1)
    images2 = np.array(image2)

    print(sewar.ssim(images1, images2)[0])
    print(ssim(images1, images2)[0])
