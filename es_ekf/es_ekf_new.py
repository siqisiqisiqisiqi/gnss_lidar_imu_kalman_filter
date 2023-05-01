# Main code for the Coursera SDC Course 2 final project
#
# Author: Trevor Ablett
# University of Toronto Institute for Aerospace Studies

# Assignments Solution Author: Engin Bozkurt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import Bounds
from rotations import Quaternion, skew_symmetric

pi = 3.1415926

# DataLoader
with open('data/custom_data/2023-03-06-12-17-56_8.csv', 'rb') as file:
    data = pd.read_csv(file)
    pos_data = data.to_numpy()

with open('data/custom_data/2023-03-06-12-17-56_8_lidar.csv', 'rb') as file:
    lidar_data = pd.read_csv(file)
    lidar_pos_data = lidar_data.to_numpy()
################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
GPS_length = np.count_nonzero(~np.isnan(data['GPS_t'].to_numpy()))
IMU_length = np.count_nonzero(~np.isnan(data['IMU_t'].to_numpy()))
LIDAR_length = np.count_nonzero(~np.isnan(lidar_data['t'].to_numpy()))

imu_f = {'acc': np.array(
    [data['linear_acc_x'], data['linear_acc_y'], data['linear_acc_z']])}
imu_w = {'ang_vel': np.array(
    [data['angular_vel_x'], data['angular_vel_y'], data['angular_vel_z']])}
gnss = {'gnss': np.array([data['x'], data['y'], data['z']])[:, 1:GPS_length]}
lidar = {'lidar': np.array(
    [lidar_data['x'], lidar_data['y'], lidar_data['z']])}
gt_data = gnss

imu_t = data['IMU_t'].to_numpy()
gnss_t = data['GPS_t'].to_numpy()[1:GPS_length]
lidar_t = lidar_data['t'].to_numpy()
gnss_t = gnss_t.squeeze()
imu_t = imu_t.squeeze()

start_t = min(gnss_t[0], imu_t[0], lidar_t[0])
gnss_t = (gnss_t - start_t)
imu_t = (imu_t - start_t)
lidar_t = (lidar_t - start_t)

################################################################################################
# lidar calibration and imu calibration
# add noise to the RTK gnss data
################################################################################################
# add noise to the gnss data
noise = np.random.normal(0, 0.1, size=(3, len(gnss_t)))
gnss["gnss"] = gnss["gnss"] + noise

# lidat calibration
w, h = lidar_data.shape
bounds = Bounds([-0.2, -0.2, 0, -0.2, -0.2, -1],
                [0.2, 0.2, 2 * pi, 0.2, 0.2, 1])
w, h = lidar_data.shape


def lidar_calibration(x):
    """The calibration error function"""
    q0 = Quaternion(euler=(x[0], x[1], x[2])).to_numpy()
    C_li = Quaternion(*q0).to_mat()
    t_li_i = np.array([x[3], x[4], x[5]])
    lidar_data = (C_li @ lidar['lidar']).T + t_li_i
    return np.linalg.norm(lidar_data - gnss["gnss"].T) / (w * h)


# x0 = [0, 0, 0, 0, 0, 0]
# res = minimize(lidar_calibration, x0, method='trust-constr',
#                options={'verbose': 1}, bounds=bounds)
# print(f"lidar calibration value is {res.x}")
# x = [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5]]


x = [-0.02361067, -0.05826697, 0.08551143, 0.19998565, 0.19998235, 0.05967025]
q0 = Quaternion(euler=(x[0], x[1], x[2])).to_numpy()
C_li = Quaternion(*q0).to_mat()
t_li_i = np.array([x[3], x[4], x[5]])
lidar_data = (C_li @ lidar['lidar']).T + t_li_i

# imu calibration
# acc_offset = np.array([[0.76056858, 0.35444493, 9.58094738 - 9.81]])
acc_offset = np.array([[0.76056858, 0.35444493, 9.58094738 - 9.9]])
imu_f['acc'] = (imu_f['acc'].T - acc_offset).T

################################################################################################
# start the kalman filter
################################################################################################

# Init Constants
var_imu_f = 0.01
var_imu_w = 0.01
var_gnss = 0.2
var_lidar = 0.1
gravity = 9.81
g = np.array([0, 0, -gravity])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

# Initial Values
gt = np.zeros([pos_data.shape[0], 3])
p_est = np.zeros([pos_data.shape[0], 3])  # position estimates
v_est = np.zeros([pos_data.shape[0], 3])  # velocity estimates
# orientation estimates as quaternions
q_est = np.zeros([pos_data.shape[0], 4])
# covariance matrices at each timestep
p_cov = np.zeros([pos_data.shape[0], 9, 9])

# Set initial values
x0 = gnss['gnss'][0, 0]
y0 = gnss['gnss'][1, 0]
z0 = gnss['gnss'][2, 0]

p_est[0] = np.array([x0, y0, z0])
q_est[0] = Quaternion(euler=(0, 0, 0)).to_numpy()
p_cov[0] = np.eye(9)  # covariance of estimate

# Measurement Update


def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # Compute Kalman Gain
    R_cov = sensor_var * np.eye(3)
    K = p_cov_check.dot(h_jac.T.dot(np.linalg.inv(
        h_jac.dot(p_cov_check.dot(h_jac.T)) + R_cov)))
    # Compute error state
    delta_x = K.dot(y_k - p_check)
    # Correct predicted state
    p_check = p_check + delta_x[:3]
    v_check = v_check + delta_x[3:6]
    q_check = Quaternion(axis_angle=delta_x[6:]).quat_mult(q_check)
    # Compute corrected covariance
    p_cov_check = (np.eye(9) - K.dot(h_jac)).dot(p_cov_check)

    return p_check, v_check, q_check, p_cov_check


# Main Filter Loop
# start at 1 b/c we have initial prediction from gt
rtk_data = 0
for k in range(1, imu_f['acc'].shape[1]):
    delta_t = imu_t[k] - imu_t[k - 1]
    # Update nominal state with IMU inputs
    Rotation_Mat = Quaternion(*q_est[k - 1]).to_mat()

    p_est[k] = p_est[k - 1] + delta_t * v_est[k - 1] + 0.5 * \
        (delta_t ** 2) * (Rotation_Mat.dot(imu_f['acc'][:, k - 1]) + g)
    v_est[k] = v_est[k - 1] + delta_t * \
        (Rotation_Mat.dot(imu_f['acc'][:, k - 1]) + g)
    q_est[k] = Quaternion(euler=delta_t * imu_w['ang_vel']
                          [:, k - 1]).quat_mult(q_est[k - 1])

    # Linearize Motion Model and compute Jacobians
    F = np.eye(9)
    imu = imu_f['acc'][:, k - 1].reshape(3)
    F[0:3, 3:6] = delta_t * np.eye(3)
    F[3:6, 6:9] = Rotation_Mat.dot(-skew_symmetric(imu)) * delta_t

    # Propagate uncertainty
    Q = np.eye(6)
    Q[0:3, 0:3] = var_imu_f * Q[0:3, 0:3]
    Q[3:6, 3:6] = var_imu_w * Q[3:6, 3:6]
    # Integration acceleration to obstain Position
    Q = (delta_t ** 2) * Q
    p_cov[k] = F.dot(p_cov[k - 1]).dot(F.T) + l_jac.dot(Q).dot(l_jac.T)

    # 3. Check availability of GNSS and LIDAR measurements

    for i in range(len(gnss_t)):
        if abs(gnss_t[i] - imu_t[k]) < 0.01:
            rtk_data = gt_data['gnss'][:, i]
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss['gnss'][:, i], p_est[k],
                                                                        v_est[k], q_est[k])

    for i in range(len(lidar_t)):
        if abs(lidar_t[i] - imu_t[k]) < 0.01:
            p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar, p_cov[k],
                                                                        lidar_data[i], p_est[k], v_est[k], q_est[k])
    # assign the ground truth
    gt[k, :] = rtk_data

# Results and Analysis
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2], label='Estimated')
ax.plot(gnss['gnss'][0, :], gnss['gnss'][1, :],
        gnss['gnss'][2, :], label='Gnss with noise')
ax.plot(lidar_data[:, 0], lidar_data[:, 1],
        lidar_data[:, 2], label='lidar data')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Estimated Trajectory')
ax.legend()
ax.set_zlim(-5, 5)
plt.show()

# ################################################################################################
# # We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# # included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# # The uncertainty bounds are +/- 3 standard deviations based on our uncertainty.
# ################################################################################################

# error_fig, ax = plt.subplots(2, 3)
# error_fig.suptitle('Error plots')
# num_gt = gt.shape[0]
# p_est_euler = []

# # Convert estimated quaternions to euler angles
# for q in q_est:
#     p_est_euler.append(Quaternion(*q).to_euler())
# p_est_euler = np.array(p_est_euler)

# # Get uncertainty estimates from P matrix
# p_cov_diag_std = np.sqrt(np.diagonal(p_cov, axis1=1, axis2=2))

# titles = ['x', 'y', 'z', 'x rot', 'y rot', 'z rot']


# for i in range(3):
#     ax[0, i].plot(range(num_gt), gt[:, i] - p_est[:num_gt, i])
#     ax[0, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i], 'r--')
#     ax[0, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i], 'r--')
#     ax[0, i].set_title(titles[i])

# for i in range(3):
#     ax[1, i].plot(range(num_gt), p_est_euler[:num_gt, i])
#     ax[1, i].plot(range(num_gt), 3 * p_cov_diag_std[:num_gt, i+6], 'r--')
#     ax[1, i].plot(range(num_gt), -3 * p_cov_diag_std[:num_gt, i+6], 'r--')
#     ax[1, i].set_title(titles[i+3])
# plt.show()
