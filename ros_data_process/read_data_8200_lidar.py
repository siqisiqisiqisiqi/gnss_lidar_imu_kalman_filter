#!/usr/bin/env python
import rosbag
import numpy as np
import pandas as pd
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import time

save = True


def read_bag_postion(filepath):

    file_path = '/'.join(filepath.split('/')[:-1])
    bag = rosbag.Bag(filepath)
    index = 0
    lidar_time = []
    for topic, msg, t in bag.read_messages(topics='/os_cloud_node/points'):
        current_lidar_time = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
        lidar_time.append(current_lidar_time)

        t = time.time()

        # self.lock.acquire()
        gen = pc2.read_points(msg, skip_nans=True)
        int_data = list(gen)

        xyz = np.empty((len(int_data), 3))
        rgb = np.empty((len(int_data), 3))
        idx = 0
        for x in int_data:
            xyz[idx, :] = x[:3]
            idx = idx + 1

        if save == True:
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(xyz)
            # out_pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
            o3d.io.write_point_cloud(
                f"{file_path}/lidar3/{index}.ply", out_pcd)
        index = index + 1

    df = pd.DataFrame(lidar_time, columns=['lidar_t'])
    df.to_csv(f'{file_path}/lidar3/lidar_t3.csv', index=False)


if __name__ == '__main__':
    fp = 'RTK_DATA/LiDAR_RTK/2023-03-06-12-17-56_8.bag'
    read_bag_postion(fp)
