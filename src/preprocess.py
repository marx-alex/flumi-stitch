import numpy as np
import cv2 as cv
import warnings


def preprocess(file, scale=None):
    """Loads the microscope image.

    Args:
        file (str): file path
        scale (float): scaling factor, <1 means down-resolutioning
            if None, no scaling

    Returns:
        numpy.ndarray [height, width]: loaded grayscale image
        affine.Affine: the transform from the loaded image to the
            original image
    """
    if scale is None:
        skip_scale = True
        scale = 1
    else:
        skip_scale = False

    # read original image metadata
    with warnings.catch_warnings():
        img = cv.imread(file, cv.IMREAD_ANYDEPTH)
        raw_width, raw_height = img.shape[1], img.shape[1]

    if not skip_scale:
        out_width, out_height = int(round(raw_width * scale)), int(round(raw_height * scale))
        # compute transform and output size
        src = np.array([[0, 0],
                        [out_width, 0],
                        [0, out_height]], dtype=np.float32)

        dst = np.array([[0, 0],
                        [raw_width, 0],
                        [0, raw_height]], dtype=np.float32)

        out_trans = cv.getAffineTransform(src, dst)
        out_trans = np.concatenate((out_trans, np.array([[0, 0, 1]])), axis=0)
        out_trans_inv = cv.getAffineTransform(dst, src)
        out_trans_inv = np.concatenate((out_trans_inv, np.array([[0, 0, 1]])), axis=0)

        # resize and store pixels as uint8
        max_value = float(np.iinfo(img.dtype).max)
        img = cv.convertScaleAbs(cv.resize(img, dsize=(out_width, out_height)), alpha=(255.0 / max_value))

        return img, out_trans, out_trans_inv

    return img
