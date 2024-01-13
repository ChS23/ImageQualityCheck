import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


def ssim(
        img1: np.ndarray,
        img2: np.ndarray,
        K: tuple = None,
        window=None,
        L: float = None
) -> tuple:
    def _fspecial_gauss(size, sigma):
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()

    def _matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def conv2(x, y, mode='same'):
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

    def filter2(x, y, mode='same'):
        return conv2(x, np.rot90(y, 2), mode=mode)

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

        # window = gaussian_filter(np.ones((11, 11)), 1.5)
        # window = _fspecial_gauss(11, 1.5)
        window = _matlab_style_gauss2D((11, 11), 1.5)
        K = (0.01, 0.03)
        L = 255

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    f = max(1, round(min(M, N) / 256))

    if f > 1:
        lpf = np.ones((f, f)) / f ** 2

        # Необходимо заменить на python аналог
        # img1 = imfilter(img1,lpf,'symmetric','same');
        # img2 = imfilter(img2,lpf,'symmetric','same');
        img1 = convolve2d(img1, lpf, mode='same', boundary='symm')
        img2 = convolve2d(img2, lpf, mode='same', boundary='symm')

        img1 = img1[::f, ::f]
        img2 = img2[::f, ::f]

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    window = window / np.sum(np.sum(window))
    # ssim_map = convolve(img1, window, mode='reflect')
    # w1 = convolve(img2, window, mode='reflect')

    # Необходимо заменить на python аналог
    ssim_map = filter2(window, img1, 'valid')
    # ssim_map = convolve2d(window, img1, mode='valid')
    w1 = filter2(window, img2, 'valid')
    w2 = ssim_map * w1
    w2 = 2 * w2 + C1
    w1 = (w1 - ssim_map) ** 2 + w2
    # ssim_map = convolve(img1 * img2, window, mode='reflect')
    ssim_map = filter2(window, img1 * img2, 'valid')
    ssim_map = (2 * ssim_map + (C1 + C2)) - w2
    ssim_map *= w2
    img1 = img1 ** 2
    img2 = img2 ** 2
    img1 = img1 + img2

    if C1 > 0 and C2 > 0:
        # w2 = convolve(img1, window, mode='reflect')
        w2 = filter2(window, img1, 'valid')
        w2 = w2 - w1 + (C1 + C2)
        w2 = w2 * w1
        ssim_map /= w2
    else:
        # w3 = convolve(img1, window, mode='reflect')
        w3 = filter2(window, img1, 'valid')
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
