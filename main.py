import time
import os
import yaml
import cv2 as cv

from src import plate, stitch


def run(cfg):
    """Runs the main script

    Args:
        cfg (dict): all configurations
        verbose (bool)

    Returns:
        src.vrt.VirtualRaster: the virtual raster built
    """
    tic = time.time()

    # step 1: create virtual raster
    p = plate.PlateData.from_dir(
        img_dir=cfg['in_dir_images'],
        out_dir=cfg['out_dir_images'],
        meta_dir=cfg['meta_dir'],
        img_suffix=cfg['img_suffix'],
        img_meta=cfg['img_meta'],
        rows=cfg['n_rows'],
        cols=cfg['n_cols'],
        reading=cfg['reading'],
        overlap=cfg['overlap'],
        channels=cfg['channels'],
        feat_channel=cfg['feat_channel'],
        jupyter=cfg['jupyter'],
        verbose=cfg['verbose'])

    # step 2: build stitcher
    s = stitch.Stitcher(
        scales=cfg['scales'],
        lowe_ratio=cfg['lowe_ratio'],
        min_inliers=cfg['min_inliers'],
        ransac_reproj_threshold=cfg['ransac_reproj_threshold'])

    # step 3: build graph
    p.build_graph(verbose=cfg['verbose'])

    # for building links and optimization iterate over wells
    for well, _ in p.df.groupby('well'):
        print(f'Start Stitching Well {well}')

        # step 4: build links
        p.build_links(
            func=s.stitch_pair,
            well=well,
            approach=cfg['approach'],
            verbose=cfg['verbose'])

        # step 5: globally optimize
        p.global_optimize(
            well=well,
            logging=cfg['logging'],
            logdir=cfg['meta_dir'],
            n_iter=cfg['optim_n_iter'],
            lr_xy=cfg['optim_lr'],
            lr_scheduler_milestones=cfg['optim_lr_scheduler_milestones'],
            output_iter=cfg['output_iter'],
            verbose=cfg['verbose'])

        # step 6: apply optimized transforms to all channels
        # and write stitched images
        for channel in p.channels:

            p.image_composition(
                well=well,
                channel=channel,
                method=cfg['composition'])

            # save stitched image
            if p.output is not None:
                cv.imwrite(os.path.join(
                    cfg['out_dir_images'],
                    f"{well} - {channel}{cfg['img_suffix']}"),
                    p.output)

        print(f'Done Stitching Well {well}')

    # step 7: write xml file with metadata
    if p.meta_dir is not None:
        p.write_meta()

    toc = time.time()
    print(f'Main function runtime: {int((toc - tic) / 60):d} minute(s) and {(toc - tic) % 60:0.3f} second(s)')
    return p


if __name__ == '__main__':
    # parse config file
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # run through all the steps
    p = run(cfg)
