#!/usr/bin/env python3
import rosbag
import csv
import numpy as np
import pandas as pd

def read_bag_postion(filepath):
    bag = rosbag.Bag(filepath)
    position = []
    heading = []
    imu = []
    title = ['x', 'y', 'z', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'angular_vel_x',
             'angular_vel_y', 'angular_vel_z']
    file_path = '/'.join(filepath.split('/')[:-1])
    filename = filepath.split('/')[-1].split('.')[0]

    # Get the position info from odom message
    for topic, msg, t in bag.read_messages(topics='/novatel/oem7/odom'):
        position.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9])
    # Normalize the odom message
    x0, y0, z0, _ = position[0]
    norm_position = []
    for i, pose in enumerate(position):
        norm_position.append([pose[0] - x0, pose[1] - y0, pose[2] - z0, pose[3]])
    norm_position = np.array(norm_position)
    for topic, msg, t in bag.read_messages(topics='/os_cloud_node/imu'):
        imu.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9])
    imu = np.array(imu)

    df1 = pd.DataFrame(norm_position, columns=['x', 'y', 'z', 'GPS_t'])
    df2 = pd.DataFrame(imu, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'IMU_t'])
    file = pd.concat([df1, df2], axis=1)
    file.to_csv(f'{file_path}/{filename}.csv', index=False)

if __name__ == '__main__':
    fp = 'lidar/2023-03-06-12-18-49_9.bag'
    read_bag_postion(fp)
