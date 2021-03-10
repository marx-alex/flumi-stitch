import numpy as np
import cv2 as cv
from skimage.registration import phase_cross_correlation
from skimage.filters import difference_of_gaussians

from .preprocess import preprocess

# seeding
cv.setRNGSeed(0)


class Stitcher(object):
    """Stitches images together.

    Args:
        scales (list of float): scaling factor for stitching
            this should be a list and low-res images will be used first
            if no match is found, higher-res images will be used
        lowe_ratio (float): Lowe's ratio for discarding false matches
            the lower, the more false matches are discarded, defaults to 0.7
        min_inliers (int): minimum number of matches to attempt
            estimation of affine transform, the higher, the more high-quality
            the match, defaults to 50, this is also used for checking whether
            a higher resolution image should be used, higher res matching
            is attempted when no. of inliers from RANSAC < min_inliers
            minimum number of inliers is 4
        ransac_reproj_threshold (float): max reprojection error in RANSAC
            to consider a point as an inlier, the higher, the more tolerant
            RANSAC is, defaults to 7.0
    """

    def __init__(self, scales=None, lowe_ratio=0.7, min_inliers=4,
                 ransac_reproj_threshold=7.0):
        # store preprocessings params
        if scales is None:
            print('No scale defined. Use scales to speed up keypoint detection')
        self.scales = [1] if scales is None else scales
        # create feature detector
        self.fd = cv.SIFT_create()
        # create feature matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)  # or pass empty dictionary
        self.mt = cv.FlannBasedMatcher(index_params, search_params)
        # Lowe's ratio for discarding false matches
        self.lowe_ratio = lowe_ratio
        # minimum feature matches to attempt transform estimation
        # if RANSAC inliers < min_inliers, higher resolution images are used
        self.min_inliers = min_inliers
        # RANSAC reprojection threshold
        # maximum reprojection error in the RANSAC algorithm
        # to consider a point as an inlier
        self.ransac_reproj_threshold = ransac_reproj_threshold

    def feature_affine(self, img0, img1, masks=None,
                       verbose=False):
        """Estimates the affine transformation.

        Args:
            img0, img1 (numpy.ndarray [height, width]): input images
            masks (tuple): masks for keypoint detection
            verbose (bool): return additional diagnostic values

        Returns:
            affine.Affine or NoneType: affine transform to fit
                img1 onto img0, None if no match is found
            dict: diagnostics (if verbose)
        """
        if masks is not None:
            assert len((img0, img1)) == len(masks), 'Incorrect number of masks'

            # unpack masks
            mask0, mask1 = masks

            # detect features, compute descriptors with mask
            kp0, des0 = self.fd.detectAndCompute(img0, mask=mask0)
            kp1, des1 = self.fd.detectAndCompute(img1, mask=mask1)

        else:
            # detect features, compute descriptors
            kp0, des0 = self.fd.detectAndCompute(img0, mask=None)
            kp1, des1 = self.fd.detectAndCompute(img1, mask=None)

        # match descriptors
        matches = self.mt.knnMatch(des0, des1, k=2)  # query, train
        # store all the good matches as per Lowe's ratio test
        good = []

        for m0, m1 in matches:
            if m0.distance < self.lowe_ratio * m1.distance:
                good.append(m0)

        if verbose:
            diag = {'n_match': len(good)}

        # with all good matches, estimate affine transform w/ RANSAC
        if len(good) > self.min_inliers:
            pts0 = np.array([kp0[m.queryIdx].pt for m in good])
            pts1 = np.array([kp1[m.trainIdx].pt for m in good])
            transform, inliers = cv.estimateAffinePartial2D(
                pts0, pts1,
                method=cv.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold)

            transform = np.concatenate((transform, np.array([[0, 0, 1]])), axis=0)

            if verbose:
                diag['n_inlier'] = inliers.sum()

            if inliers.sum() < self.min_inliers:
                return (None, diag) if verbose else None

            return (transform, diag) if verbose else transform
        else:
            return (None, diag) if verbose else None

    def fourier_affine(self, img0, img1,
                       masks=None, verbose=False):
        """Estimate translations by masked normalized
        cross-correlation after fourier transformation

        img0, img1 (numpy.ndarray [height, width]): input images
        masks (tuple): masks for keypoint detection
        verbose (bool): return additional diagnostic values

        Returns:
            np.ndarray or NoneType: affine transform to fit
                img1 onto img0, None if RMSE is > 0.4
            dict: diagnostics (if verbose)
        """
        # preprocess the images
        # First, band-pass filter both images
        low_sigma = 1
        high_sigma = None
        img0_dog = difference_of_gaussians(img0, low_sigma=low_sigma, high_sigma=high_sigma)
        img1_dog = difference_of_gaussians(img1, low_sigma=low_sigma, high_sigma=high_sigma)

        # find translations masked on non-masked
        if masks is not None:
            assert len((img0, img1)) == len(masks), 'Incorrect number of masks'
            mask0, mask1 = masks
            shifts = phase_cross_correlation(img1_dog, img0_dog,
                                             reference_mask=mask1, moving_mask=mask0)
            y_shift, x_shift = shifts[:2]
            if verbose:
                diag = {'low_sigma': low_sigma, 'high_sigma': high_sigma,
                        'y_shift_scaled': y_shift, 'x_shift_scaled': x_shift}

        else:
            shifts, error, phasediff = phase_cross_correlation(img1_dog, img0_dog,
                                                               upsample_factor=10)
            y_shift, x_shift = shifts[:2]

            if verbose:
                diag = {'low_sigma': low_sigma, 'high_sigma': high_sigma,
                        'y_shift_scaled': y_shift, 'x_shift_scaled': x_shift,
                        'RMSE_shift': error, 'phasediff_shift': phasediff}

            # if error > 0.4 result is not acceptable
            if error > 0.4:
                return (None, diag) if verbose else None

        translation = np.array([[1, 0, x_shift],
                                [0, 1, y_shift],
                                [0, 0, 1]], dtype=np.float32)

        return (translation, diag) if verbose else translation

    def stitch_pair(self, img0, img1, pos, approach='feature_based',
                    overlap=None, verbose=False, **kwargs):
        """Stitch images together using a pyramid of resolutions.

        Args:
            img0, img1 (str): file path
            pos (tuple of two tuples of ints): grid positions of images (row, column)
            approach (str):
                'feature-based': finds translation with features in images
                'fourier-based': finds translation by cross-correlation of fourier
                transformed images
                'feature-else-fourier': if no translation is found by feature-based
                approach, use fourier-based approach
            overlap (float): overlap of images
            verbose (bool)
            **kwargs: passed to estimate_affine()

        Returns:
            affine.Affine or NoneType: translation to fit
                img1 onto img0, None if no match is found
                the relative transform is in terms of the original images
            dict: diagnostics (if verbose)
        """
        # iterate over resolutions
        for scale in self.scales:
            img0_array, trans0, trans0_inv = preprocess(img0, scale=scale)
            img1_array, trans1, trans1_inv = preprocess(img1, scale=scale)

            # get mask depending on position of field
            mask0, mask1 = self.get_roimask(dsize=img0_array.shape, pos=pos, overlap=overlap)

            assert approach in ['feature-based', 'fourier-based', 'feature-else-fourier'], \
                'approach unknown'

            if approach == 'feature-based':
                output = self.feature_affine(
                    img0_array, img1_array, masks=(mask0, mask1), verbose=verbose, **kwargs)
            elif approach == 'fourier-based':
                output = self.fourier_affine(
                    img0_array, img1_array, masks=(mask0, mask1), verbose=verbose)
            elif approach == 'feature-else-fourier':
                output = self.feature_affine(
                    img0_array, img1_array, masks=(mask0, mask1), verbose=verbose, **kwargs)

                if verbose:
                    relative_trans, diag = output
                    diag['img0'] = img0
                    diag['img1'] = img1
                    diag['scale'] = scale
                    print(diag)
                else:
                    relative_trans = output
                # use fourier-based approach if output is None
                if relative_trans is None:
                    output = self.fourier_affine(
                        img0_array, img1_array, masks=(mask0, mask1), verbose=verbose)

            if verbose:
                relative_trans, diag = output
                diag['img0'] = img0
                diag['img1'] = img1
                diag['scale'] = scale
                print(diag)
            else:
                relative_trans = output
            # return if transform is found
            if relative_trans is not None:
                # take into account the transforms between the original images
                # and the preprocessed images
                trans = np.matmul(np.matmul(trans1, relative_trans), trans1_inv)
                # take x and y shift and return translation parameters
                x_shift, y_shift = trans[0, 2], trans[1, 2]
                translation = np.array([x_shift, y_shift])
                return translation

        return None

    @staticmethod
    def get_roimask(dsize, pos, overlap):
        """Finds mask depending on position of images.

        Args:
            dsize (tuple of ints): size of images
            pos (tuple of two tuples of ints): grid positions of images (row, column)
            overlap (float): overlap of images in percentage

        Returns:
            numpy.array: masks to find keypoints on original images
        """
        # function needs exactly two positions
        assert len(pos) == 2, 'Mask for keypoint detection needs two image positions'

        if overlap is None:
            overlap = 1.

        # unpack positions
        pos0, pos1 = pos
        row0, col0 = pos0
        row1, col1 = pos1

        # make mask of size dsize
        mask0 = np.zeros(dsize, dtype=np.uint8)
        mask1 = np.zeros(dsize, dtype=np.uint8)

        if row1 == row0:
            if col1 > col0:
                y_min0, y_max0, x_min0, x_max0 = 0, dsize[0], int(dsize[1] * (1 - overlap)), dsize[1]
                y_min1, y_max1, x_min1, x_max1 = 0, dsize[0], 0, int(dsize[1] * overlap)
            if col1 < col0:
                y_min0, y_max0, x_min0, x_max0 = 0, dsize[0], 0, int(dsize[1] * overlap)
                y_min1, y_max1, x_min1, x_max1 = 0, dsize[0], int(dsize[1] * (1 - overlap)), dsize[1]
        elif col1 == col0:
            if row1 > row0:
                y_min0, y_max0, x_min0, x_max0 = int(dsize[0] * (1 - overlap)), dsize[0], 0, dsize[1]
                y_min1, y_max1, x_min1, x_max1 = 0, int(dsize[0] * overlap), 0, dsize[1]
            if row1 < row0:
                y_min0, y_max0, x_min0, x_max0 = 0, int(dsize[0] * overlap), 0, dsize[1]
                y_min1, y_max1, x_min1, x_max1 = int(dsize[0] * (1 - overlap)), dsize[0], 0, dsize[1]
        else:
            raise ValueError('Arguments for positions incorrect')

        # make masked area max value
        mask0[y_min0:y_max0, x_min0:x_max0] = np.iinfo(mask0.dtype).max
        mask1[y_min1:y_max1, x_min1:x_max1] = np.iinfo(mask0.dtype).max

        return mask0, mask1
