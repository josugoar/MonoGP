import argparse

from .dataset_converters import kitti_converter as kitti

SEGMENT_PLANE_CONFIG = dict(
    distance_threshold=0.33, ransac_n=3, num_iterations=2000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, default='./data/kitti')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--image-ids', type=int, default=7481)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    kitti.create_planes(
        args.root_path,
        training=not args.testing,
        image_ids=args.image_ids,
        num_worker=args.workers,
        show=not args.no_show,
        **SEGMENT_PLANE_CONFIG)


if __name__ == '__main__':
    main()
