import argparse
from concurrent import futures as futures

import numpy as np
import open3d as o3d

from tools.dataset_converters import kitti_data_utils as kitti

SEGMENT_PLANE_KWARGS = {
    'distance_threshold': 0.33,
    'ransac_n': 3,
    'num_iterations': 2000
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate planes from point clouds')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/kitti',
        help='specify the root path of dataset')
    parser.add_argument(
        '--testing', action='store_true', help='use testing set or not')
    parser.add_argument(
        '--image-ids',
        type=int,
        default=7481,
        help='number of images to be processed')
    parser.add_argument(
        '--workers', type=int, default=8, help='number of threads to be used')
    parser.add_argument(
        '--not-show',
        action='store_true',
        help='do not show plane generation results')
    args = parser.parse_args()
    return args


def get_kitti_plane_info(path,
                         training=True,
                         image_ids=7481,
                         num_worker=8,
                         show=True,
                         **segment_plane_kwargs):
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        num_features = 4

        velodyne_path = kitti.get_velodyne_path(
            idx, path, training, relative_path=False)
        calib_path = kitti.get_calib_path(
            idx, path, training, relative_path=False)
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        Tr_velo_to_cam = np.array([
            float(info) for info in lines[5].split(' ')[1:13]
        ]).reshape([3, 4])
        Tr_velo_to_cam = kitti._extend_matrix(Tr_velo_to_cam)

        points = np.fromfile(
            velodyne_path, dtype=np.float32).reshape(-1, num_features)
        pcd_lidar = o3d.t.geometry.PointCloud()
        pcd_lidar.point.positions = points[:, :3]
        info['pcd_lidar'] = pcd_lidar

        pcd_cam = pcd_lidar.clone().transform(Tr_velo_to_cam)
        plane, inliers = pcd_cam.segment_plane(**segment_plane_kwargs)
        plane *= -1
        info['inliers'] = inliers

        # if not show:
        #     plane_path = kitti.get_plane_path(
        #         idx, path, training, relative_path=False)
        #     with open(plane_path, 'w') as f:
        #         print('# Matrix', file=f)
        #         print('WIDTH 4', file=f)
        #         print('HEIGHT 1', file=f)
        #         print(' '.join(map('{:.2e}'.format, plane.numpy())), file=f)

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        infos = executor.map(map_func, image_ids)

    for info in infos:
        pcd_lidar = info['pcd_lidar']
        inliers = info['inliers']

        if show:
            inlier_cloud = pcd_lidar.select_by_index(inliers)
            inlier_cloud = inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = pcd_lidar.select_by_index(inliers, invert=True)
            o3d.visualization.draw([inlier_cloud, outlier_cloud])


def main():
    args = parse_args()

    get_kitti_plane_info(
        args.root_path,
        training=not args.testing,
        image_ids=args.image_ids,
        num_worker=args.workers,
        show=not args.not_show,
        **SEGMENT_PLANE_KWARGS)


if __name__ == '__main__':
    main()
