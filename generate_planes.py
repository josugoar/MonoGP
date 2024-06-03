import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description='Extract planes from point clouds')
    parser.add_argument(
        '--points-path',
        type=Path,
        default='./data/kitti/training/velodyne/',
        help='Path to the directory containing the point clouds'
    )
    parser.add_argument(
        '--planes-path',
        type=Path,
        default='./data/kitti-test/training/planes/',
        help='Path to the directory to save the extracted planes'
    )
    args = parser.parse_args()

    for points_file in args.points_path.iterdir():
        num_features = 4
        points = np.fromfile(points_file, dtype=np.float32).reshape(
            -1, num_features)
        pcd = o3d.t.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        plane, _ = pcd.segment_plane()
        # TODO: transform plane to camera coordinates
        plane = ...

        frame_idx = points_file.stem
        plane_file = f'{args.planes_path}/{frame_idx}.txt'
        with open(plane_file, mode='w') as plane_fp:
            print('# Matrix', file=plane_fp)
            print('WIDTH 4', file=plane_fp)
            print('HEIGHT 1', file=plane_fp)
            print(' '.join(map('{:.2e}'.format, plane)), file=plane_fp)


if __name__ == '__main__':
    main()
