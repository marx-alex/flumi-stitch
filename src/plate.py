import re
import os
import warnings
import collections

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm, tqdm_notebook
import cv2 as cv
import pywt
from lxml import etree
import matplotlib.pyplot as plt
from scipy import stats

from .optim import optimize
from .preprocess import preprocess


class PlateData(object):
    """Virtual raster.

    Stores all plate-associated information.

    Args:
        df (pandas.DataFrame): contains the following keys
            'file_id': image file paths, should be concatenated with
                img_dir and a proper suffix in order to form complete
                paths
            'well': name of the well
            'field_id': number of the field (continuous)
            'row', 'column': the row and columns of a field in a well
            'channel': name of the channels
            'file_path': file paths to img files
            'width', 'height': the width and height of an image (original),
                in pixels
            'x_pos', 'y_pos': position of images projected on a composite
            'theta', 'scale': initialization parameter for optimization
        img_dir (str): directory to images
        out_dir (str): directory to output image
        meta_dir (str): directory to store metadata
        rows (int): number of rows per field
        cols (int): number of columns per field
        overlap (float): overlap of images in percentage
        width, height (int): dimensions of images
        img_depth (np.dtype): depth of image
        channels (list): channels in which images were taken
        feat_channel (str): name of the channel where the most features are expected
        jupyter (bool)
    """

    def __init__(self, df, img_dir, out_dir, meta_dir, rows,
                 cols, overlap, width, height, img_depth, channels, feat_channel, jupyter):
        print('Reading Plate Data.')
        # initialize self.df
        self.df = df
        # store paths
        assert img_dir != out_dir
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.meta_dir = meta_dir
        # store plate information
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.width = width
        self.height = height
        # store image depth
        self.img_depth = img_depth
        self.channels = channels
        self.feat_channel = feat_channel
        # initialize graph and links
        self.graph = collections.defaultdict(list)
        self.links = {}
        # true if run on jupyter notebook
        self.jupyter = jupyter
        # attributes initialized in other functions
        self.optim_iters = None
        self.optim_losses = None
        self.optim_trans = None
        self.optim_params = None
        self.output = None

    @classmethod
    def from_dir(cls, img_dir, out_dir, meta_dir, img_suffix,
                 img_meta, rows, cols, reading, overlap, channels,
                 feat_channel, jupyter, verbose):
        """Loads information about images of whole plate stored in a file.

        Args:
            img_dir (str): path to images
            out_dir (str): path for output
            meta_dir (str): path for metadata
            img_suffix (str): suffix of images
            img_meta (str): regular expression for file names
            rows, cols (int): number of rows/columns of image grids in wells
            reading (str): method in witch the fields were recorded
                'left-to-right': from left to right and top to bottom
                'top-to-bottom' from top to bottom and left to right
                'snake': snake-like direction from top to bottom
            overlap (float): overlap of images in percent
            channels (list of str): names of channels that have been used
            feat_channel (str): name of the channel where the most features are expected
            jupyter (bool)
            verbose (bool)

        """
        # compile regular expression to parse filenames
        pattern = re.compile(str(img_meta + img_suffix))
        # check for file existence
        assert os.path.exists(img_dir), 'Input path does not exist'
        assert os.path.exists(out_dir), 'Output path does not exist'
        if not os.path.exists(meta_dir) or meta_dir is None:
            meta_dir = None
            raise Exception("Metadata won't be stored!")
        # query and store file_ids, well names and field_ids
        file_ids = []
        wells = []
        field_ids = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                match = pattern.match(file)
                if pattern.match(file):
                    file_ids.append(file)
                    wells.append(str(match.group('Row')) + str(match.group('Column')))
                    field_ids.append(int(match.group('Field')))

        assert len(file_ids) != 0, f'No images that match pattern: {pattern}'
        # create a data frame with all image information
        df = pd.DataFrame({'file_id': file_ids, 'well': wells, 'field_id': field_ids})

        # add row and columns of field to the data frame depending on the reading method
        assert reading in ['left-to-right', 'top-to-bottom', 'snake'], \
            f'Reading method unknown: {reading}'
        assert rows * cols == df.field_id.max(), f'Number of Rows and Columns incorrect: ' \
                                                 f'Rows: {rows}, Columns: {cols}, Fields: {df.field_id.max()}'

        # create default column and row list
        def_col_ls = list(range(1, cols + 1)) * rows
        def_row_ls = [row for row in range(1, rows + 1) for _ in range(cols)]

        if reading == 'left-to-right':
            row_ls = def_row_ls
            col_ls = def_col_ls

        if reading == 'top-to-bottom':
            row_ls = list(range(1, rows + 1)) * cols
            col_ls = [col for col in range(1, cols + 1) for _ in range(rows)]

        if reading == 'snake':
            row_ls = [row for row in range(1, rows + 1) for _ in range(cols)]
            col_ls = (list(range(1, cols + 1)) + list(range(1, cols + 1))[::-1]) * (rows // 2)
            if len(col_ls) == 0:
                col_ls = list(range(1, cols + 1))
            elif (rows % 2) != 0:
                col_ls = col_ls + list(range(1, cols + 1))

        def field_pos(field_id, row_ls, col_ls):
            return list(zip(row_ls, col_ls))[int(field_id) - 1]

        df['row'] = df.field_id.apply(lambda x: field_pos(x, row_ls, col_ls)[0])
        df['column'] = df.field_id.apply(lambda x: field_pos(x, row_ls, col_ls)[1])

        # overwrite field_id with default ids (left-to-right)
        if reading in ['top-to-bottom', 'left-to-right']:
            field_ls = list(range(1, (cols * rows)+1))
            field_dic = {(row, col): field_id for field_id, row, col in
                         list(zip(field_ls, def_row_ls, def_col_ls))}
            df.field_id = df.apply(lambda x: field_dic[(x['row'], x['column'])], axis=1)

        # extract channel from file_id
        def find_channel(str, channels):
            for ch in channels:
                if ch in str:
                    return ch
            return np.nan

        df['channel'] = df.file_id.apply(lambda x: find_channel(x, channels))

        # drop missing values
        if verbose:
            print(f"Files without a channel from the given channel list: "
                  f"{df.file_id[df.isnull().any(axis=1)].tolist()}")

        df = df.dropna()

        # extract complete file paths
        df['file_path'] = df.file_id.apply(lambda x: os.path.join(img_dir, x))

        # query and store height and width from image metadata
        width = []
        height = []
        depth = None
        # iterate over each file
        for i, file in enumerate(df.loc[:, 'file_path'].tolist()):
            # get image depth of first image
            if i == 0:
                depth = cv.imread(file, cv.IMREAD_ANYDEPTH).dtype
            with Image.open(file) as ds:
                width.append(ds.width)
                height.append(ds.width)
        # update dataframe
        df.loc[:, 'width'] = width
        df.loc[:, 'height'] = height

        # check if heights, widths and dtypes are the same for all images
        assert len(set(df.width.tolist())) == 1, f'Dimensions are not equal: {set(df.width.tolist())}'
        assert len(set(df.height.tolist())) == 1, f'Dimensions are not equal: {set(df.height.tolist())}'
        assert depth is not None, 'Image depth can not be read'

        if verbose:
            print(f'Files that do not fit the size of the first image: '
                  f'{df.file_id[df.width != df.width[0]].tolist()}')

        # add image position to df
        def add_xy(w, h, row, col):
            x = (col - 1) * w
            y = (row - 1) * h
            return x, y

        df['x_pos'] = df.apply(lambda x: add_xy(x.width, x.height, x.row, x.column)[0], axis=1)
        df['y_pos'] = df.apply(lambda x: add_xy(x.width, x.height, x.row, x.column)[1], axis=1)

        return cls(df, img_dir=img_dir, out_dir=out_dir, meta_dir=meta_dir,
                   rows=rows, cols=cols, overlap=overlap, width=width[0], height=height[0], img_depth=depth,
                   channels=channels, feat_channel=feat_channel, jupyter=jupyter)

    def build_graph(self, verbose=False):
        """Builds an undirected graph that describes an image's neighbours.

        Args:
            verbose (bool)

        Returns:
            collections.defaultdict(list): graph
        """
        # use the first well to build a graph
        first_well = list(self.df.groupby('well'))[0][0]
        pos = list(self.df[['field_id', 'row', 'column']]
                   [(self.df.well == first_well) & (self.df.channel == self.feat_channel)]
                   .to_records(index=False))

        # match fields with neighbouring fields one-directional
        for field, row, col in pos:
            for n_field, n_row, n_col in pos:
                if n_row == row and (n_col == col + 1 or n_col == col - 1):
                    if field not in self.graph[n_field]:
                        self.graph[field].append(n_field)
                if n_col == col and (n_row == row + 1 or n_row == row - 1):
                    if field not in self.graph[n_field]:
                        self.graph[field].append(n_field)

        if verbose:
            print(f'Graph: {self.graph}')

        return self.graph

    def build_links(self, func, well, approach='feature_based', verbose=False):
        """Build links between nodes in feature channel.

        Args:
            func (function): takes two image file paths, img0 and img1
                returns an affine transformation from img1 to img0
            well (str): current well
            approach (str):
                'feature-based': finds translation with features in images
                'fourier-based': finds translation by cross-correlation of fourier
                transformed images
                'feature-else-fourier': if no translation is found by feature-based
                approach, use fourier-based approach
            verbose (bool)
        """
        # prepare pairs of indices
        # f: field, n: neighbour
        file_paths = self.df[['file_path', 'field_id']][(self.df.well == well) & (self.df.channel == self.feat_channel)]
        file_paths = file_paths.set_index('field_id').squeeze()
        # get positions of fields
        pos = self.df[['field_id', 'row', 'column']][(self.df.well == well) & (self.df.channel == self.feat_channel)]
        pos = pos.set_index('field_id')
        pairs = [(f, n, file_paths[f], file_paths[n],
                  (int(pos.loc[f, 'row']), int(pos.loc[f, 'column'])),
                  (int(pos.loc[n, 'row']), int(pos.loc[n, 'column'])))
                 for f, ns in self.graph.items() for n in ns]

        # estimate transforms for every pair
        if self.jupyter:
            result = [((f, n), func(f_path, n_path, pos=(f_pos, n_pos),
                                    approach=approach, overlap=self.overlap, verbose=verbose))
                      for f, n, f_path, n_path, f_pos, n_pos in
                      tqdm_notebook(pairs, desc=f'Building links for Well {well}')]
        else:
            result = [((f, n), func(f_path, n_path, pos=(f_pos, n_pos),
                                    approach=approach, overlap=self.overlap, verbose=verbose))
                      for f, n, f_path, n_path, f_pos, n_pos in
                      tqdm(pairs, desc=f'Building links for well {well}')]

        # optimize pairwise relative translations
        result = self.trans_optimize(result, pos, verbose=verbose)

        # collect into dictionary
        self.links.update(dict(result))

        # make links symmetric
        for f, ns in self.graph.items():
            for n in ns:
                if f < n:
                    self.links[(n, f)] = (None if self.links[(f, n)] is None
                                          else -self.links[(f, n)])

        if verbose:
            print('Links: ', self.links)

    @staticmethod
    def trans_optimize(trans, pos, thresh=3, verbose=False):
        """Optimize a list of pairwise translations

        Args:
            trans (list): List of pairwise translations
            pos (pandas.DataFrame): Dataframe with Field IDs as Index
                and Row and Column as column
            thresh (int): threshold to detect outliers
                (thresh times standard deviation)
            verbose (bool)

        Returns:
            list: Optimized pairwise translations
        """
        if verbose:
            print(f'Translations before optimization: {trans}')

        # Sort by vertical and horizontal translations
        v = []
        h = []
        for (f, n), t in trans:
            if t is not None:
                if int(pos.loc[f, 'row']) == int(pos.loc[n, 'row']):
                    h.append(t)
                else:
                    v.append(t)

        # stack lists
        v = np.stack(v)
        h = np.stack(h)

        # calculate z-scores for v and h
        vz = np.abs(stats.zscore(v, axis=0))
        hz = np.abs(stats.zscore(h, axis=0))
        # if std is 0 z-scores become np.nan
        # replace nan by 0
        vz = np.nan_to_num(vz)
        hz = np.nan_to_num(hz)

        # delete outliers from translations
        # outlier is defined to be outside of third
        # standard deviation
        v = v[((vz > -thresh) & (vz < thresh)).all(axis=1)]
        h = h[((hz > -thresh) & (hz < thresh)).all(axis=1)]
        assert v.size != 0 and h.size != 0, \
            'Translation optimization not possible, to many outliers!'

        # get mean, max and min translations for both directions
        v_mean, v_std = v.mean(axis=0), v.std(axis=0)
        v_min, v_max = v.min(axis=0), v.max(axis=0)
        h_mean, h_std = h.mean(axis=0), h.std(axis=0)
        h_min, h_max = h.min(axis=0), h.max(axis=0)

        if verbose:
            print(f'Mean horizontal translation: {h_mean}, '
                  f'Mean vertical translation: {v_mean}')
            print(f'Standard deviation for horizontal translations: {h_std}, '
                  f'Standard deviation for vertical translations: {v_std}')

        # iterate again over translations and replace
        # outliers and Nones with mean values
        optim_trans = []
        for (f, n), t in trans:
            # horizontal translations
            if int(pos.loc[f, 'row']) == int(pos.loc[n, 'row']):
                if t is None or (t < h_min).any() or (t > h_max).any():
                    optim_trans.append(((f, n), h_mean))
                else:
                    optim_trans.append(((f, n), t))
            # vertical translations
            else:
                if t is None or (t < v_min).any() or (t > v_max).any():
                    optim_trans.append(((f, n), v_mean))
                else:
                    optim_trans.append(((f, n), t))

        return optim_trans

    def global_optimize(self, well, logging=False, logdir=None, **kwargs):
        """Globally optimize to fit all images together.

        Args:
            well (str): current well
            logging (bool): prints loss graph if true
            logdir (str): path to write loss graph
            **kwargs: passed to src.optim.optimize
        """
        print('Globally optimizing.')
        # globally optimize
        iters, losses, trans, params = optimize(
            nodes=self.df.field_id[(self.df.well == well) & (self.df.channel == self.feat_channel)].tolist(),
            links=self.links,
            xs_init=self.df.x_pos[(self.df.well == well) & (self.df.channel == self.feat_channel)].tolist(),
            ys_init=self.df.y_pos[(self.df.well == well) & (self.df.channel == self.feat_channel)].tolist(),
            width=self.df.width[(self.df.well == well) & (self.df.channel == self.feat_channel)].tolist(),
            height=self.df.height[(self.df.well == well) & (self.df.channel == self.feat_channel)].tolist(),
            **kwargs)

        # collect output
        self.optim_iters = iters
        self.optim_losses = losses
        self.optim_trans = trans
        self.optim_params = params

        if logging is True:
            assert os.path.exists(logdir), 'path for logging info does not exist'
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot()
            ax.plot(self.optim_iters, self.optim_losses, c='darkred')
            ax.set_yscale('log')
            ax.set_xlabel('Iteration', fontsize=14)
            ax.set_ylabel('log loss', fontsize=14)
            ax.set_title(f'Log Loss for Well {well}', fontsize=20)
            fig.savefig(os.path.join(logdir, f'loss_{well}.png'))

    # TODO: Fix EDOF Merging
    def merge_stack(self, img0, img1):
        # store parameters as members
        wavelet_type = 'bior3.1'
        wavelet_level = 1

        stack = np.dstack((img0, img1))
        dtype = stack.dtype

        float_stack = stack.astype(np.float) / np.iinfo(dtype).max
        wavelet_coeffs = []

        # forward wavelet transformation of each z-layer
        for i, zlayer in enumerate(np.moveaxis(float_stack, -1, 0)):
            wavelet_coeffs_zlayer, S = pywt.coeffs_to_array(pywt.wavedec2(zlayer, wavelet_type, level=wavelet_level))
            wavelet_coeffs.append(wavelet_coeffs_zlayer)

        wavelet_coeffs = np.dstack(wavelet_coeffs)

        # maximum absolute value selection rule for each coefficient over z-stack
        max_indices = np.argmax(np.abs(wavelet_coeffs), axis=-1)
        ind_x, ind_y = np.indices(wavelet_coeffs.shape[:-1])
        res_wavelet_coeffs = wavelet_coeffs[ind_x, ind_y, max_indices]

        # inverse wavelet transformation of modified coefficents
        reconstructed = pywt.waverec2(pywt.array_to_coeffs(res_wavelet_coeffs, S, output_format='wavedec2'),
                                      wavelet_type)

        # reassignment step: for each pixel select nearest original grayvalue to reconstructed over z-stack
        min_indices = np.argmin(np.abs(float_stack - reconstructed[..., None]), axis=-1)
        ind_x, ind_y = np.indices(float_stack.shape[:-1])
        stack = stack[ind_x, ind_y, min_indices]

        # return corrected sample
        return stack

    def image_composition(self, well, channel, method):
        """Applies transformation to all images.

        Args:
             well (str): current well
             channel (str): current channel
             method (str): method to make composite
                'extended-depth': takes in-focus pixels from two images
                                    in overlapping regions
                'overlay': puts field with higher ids in front of fields
                            with lower ids
                'average-blending': pixel average in overlapping regions
                'linear-blending': linear gradient in overlapping regions
        """
        # return None if channel not in channel list
        if channel not in list(self.df.channel[self.df.well == well].unique()):
            print(f'Channel {channel} not found.')
            self.output = None
            return None

        # calculate output dims
        output_height = self.height * (self.rows + 1)
        output_width = self.width * (self.cols + 1)

        valid_methods = ['extended-depth', 'overlay', 'average-blending', 'linear-blending']
        assert method in valid_methods, \
            f"{method} not in {valid_methods}'"

        # make background for final composition
        background = np.zeros((output_height, output_width), dtype=self.img_depth)

        if method in ['overlay', 'average-blending', 'linear-blending']:
            final = background.copy()

            if method == 'average-blending':
                counts = np.zeros((output_height, output_width))
            if method == 'linear-blending':
                alphas = np.zeros((output_height, output_width),
                                  dtype=np.float32)

        # iterate over images in well and channel
        obj = self.df[(self.df.well == well) & (self.df.channel == channel)]
        if self.jupyter:
            df_iterrows = tqdm_notebook(
                obj.iterrows(),
                desc=f'Stitching Well {well} in Channel {channel}.',
                total=len(obj))
        else:
            df_iterrows = tqdm(
                obj.iterrows(),
                desc=f'Stitching Well {well} in Channel {channel}.',
                total=len(obj))

        for i, (r, field) in enumerate(df_iterrows):
            # load image with the preprocess function
            img = preprocess(file=field['file_path'])
            field_id = field['field_id']
            trans = self.optim_trans[-1][field_id-1]

            # update initialized values in df
            self.df.loc[(self.df.well == well) &
                        (self.df.channel == channel) &
                        (self.df.field_id == field_id),
                        'x_pos'] = self.optim_params['loc'][field_id-1][0]
            self.df.loc[(self.df.well == well) &
                        (self.df.channel == channel) &
                        (self.df.field_id == field_id),
                        'y_pos'] = self.optim_params['loc'][field_id-1][1]

            # apply shift to current image
            if method == 'overlay':
                out = final
            else:
                out = background.copy()
            out[int(trans[1]):int(trans[1]) + self.height,
                int(trans[0]):int(trans[0]) + self.width] = img

            # apply method
            if method == 'overlay':
                # alpha set to one to move out in front
                # of output
                final = out

            elif method == 'extended-depth':
                warnings.warn(f'{method} is under construction')
                if i == 0:
                    final = out
                else:
                    final = self.merge_stack(final, out)

            elif method in ['overlay', 'average-blending', 'linear-blending']:
                # compose transparency channels (alpha mask)
                alpha = np.zeros((output_height, output_width))  # nodata = 0
                alpha[int(trans[1]):int(trans[1]) + self.height,
                      int(trans[0]):int(trans[0]) + self.width] = 1

                if method == 'average-blending':
                    # alpha is divided by counts
                    # to average alpha
                    counts += alpha
                    # prevent division by zero
                    alpha = alpha / np.clip(counts, 1, None)
                    final = out * alpha + final * (1 - alpha)

                # TODO: Fix Linear-blending
                elif method == 'linear-blending':
                    warnings.warn(f'{method} is under construction')
                    # find overlapping region
                    overlap = np.logical_and(alpha, alphas)
                    # index of overlapping pixels
                    overlap_ix = np.nonzero(overlap)
                    # update alphas
                    alphas = np.logical_or(alphas, alpha)
                    # if no overlapping regions, use overlay method
                    if len(list(zip(*overlap_ix))) == 0:
                        final = out * alpha + final * (1 - alpha)
                    else:
                        x_min, x_max = overlap_ix[1].min(), overlap_ix[1].max()
                        width = (x_max - x_min) + 1
                        y_min, y_max = overlap_ix[0].min(), overlap_ix[0].max()
                        height = (y_max - y_min) + 1

                        # check if overlapping region is rectangular
                        # look for smaller bbox with zeros
                        sb = np.where(overlap[y_min:y_max + 1, x_min:x_max + 1] == 0)
                        if len(list(zip(*sb))) != 0:
                            # catch white space
                            bbox = np.min(sb[0]), np.max(sb[0]), np.min(sb[1]), np.max(sb[1])

                            # create box for horizontal overlap
                            if bbox[0] == 0:
                                hor_bbox = (y_min + bbox[1] + 1, y_max, x_min, x_max)
                            else:
                                hor_bbox = (y_min, y_min + bbox[0] - 1, x_min, x_max)
                            hor_width = (hor_bbox[3] - hor_bbox[2]) + 1
                            hor_height = (hor_bbox[1] - hor_bbox[0]) + 1

                            # create box for vertical overlap
                            if bbox[2] == 0:
                                ver_bbox = (y_min, y_max, x_min + bbox[3] + 1, x_max)
                            else:
                                ver_bbox = (y_min, y_max, x_min, x_min + bbox[2] - 1)
                            ver_width = (ver_bbox[3] - ver_bbox[2]) + 1
                            ver_height = (ver_bbox[1] - ver_bbox[0]) + 1

                            # create index grids with horizontal and
                            # vertical gradients
                            hor_grid = np.indices((hor_height, hor_width))
                            hor_grad = np.zeros((output_height, output_width))
                            hor_grad[hor_bbox[0]:hor_bbox[1] + 1, hor_bbox[2]:hor_bbox[3] + 1] = hor_grid[0]
                            ver_grid = np.indices((ver_height, ver_width))
                            ver_grad = np.zeros((output_height, output_width))
                            ver_grad[ver_bbox[0]:ver_bbox[1] + 1, ver_bbox[2]:ver_bbox[3] + 1] = ver_grid[1]

                            # add both gradient masks
                            grad = hor_grad + ver_grad

                        # if overlapping region is rectangular one
                        # gradient is enough
                        else:
                            grid = np.indices((height, width))
                            grad = np.zeros((output_height, output_width))

                            if height >= width:
                                grad[y_min:y_max + 1, x_min:x_max + 1] = grid[0]
                            elif height < width:
                                grad[y_min:y_max + 1, x_min:x_max + 1] = grid[1]

                        # create alpha mask from gradient mask
                        alpha = np.ones((output_height, output_width))
                        alpha = alpha / np.clip(grad, 1, None)
                        alpha_inv = np.zeros((output_height, output_width))
                        alpha_inv = alpha_inv / np.clip(grad, 1, None)
                        alpha_inv = 1 - alpha_inv
                        final = out * alpha_inv + final * alpha

                else:
                    raise NotImplementedError

        # crop result
        mask = final > 0
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), output_width - mask0[::-1].argmax() - 1
        row_start, row_end = mask1.argmax(), output_height - mask1[::-1].argmax() - 1
        final = final[row_start:row_end, col_start:col_end]

        self.output = final.astype(self.img_depth)

    def write_meta(self, path=None):
        """Stores metadata from self.df in xml file

        Args:
            path (str): path to store file
        """
        if path is None:
            if self.meta_dir is None:
                return None
            path = self.meta_dir

        plate_tree = etree.Element("Plate", img_dir=self.img_dir, out_dir=self.out_dir, meta_dir=path)
        for well, sub_df in self.df.groupby('well'):
            well_tree = etree.SubElement(plate_tree, "Well", name=well)  # xml.append(f"    <well id='{well}'>")
            for i, row in sub_df.iterrows():
                field_tree = etree.SubElement(well_tree, "Field", id=str(row.field_id), channel=row.channel)
                for col in row.index:
                    etree.SubElement(field_tree, f"{col}").text = str({row[col]})

        tree = etree.ElementTree(plate_tree)

        tree.write(os.path.join(path, "meta.xml"), pretty_print=True)
