import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


save = False
with open('data/custom_data/2023-03-06-12-12-06_1.csv', 'rb') as file:
    data = pd.read_csv(file)
    pos_data = data.to_numpy()

GPS_length = np.count_nonzero(~np.isnan(data['GPS_t'].to_numpy()))
gnss = {'gnss': np.array([data['x'], data['y'], data['z']])[:, 1:GPS_length]}

imu_f = {'acc': np.array(
    [data['linear_acc_x'], data['linear_acc_y'], data['linear_acc_z']])}
imu_w = {'ang_vel': np.array(
    [data['angular_vel_x'], data['angular_vel_y'], data['angular_vel_z']])}
imu_t = data['IMU_t'].to_numpy()
gnss_t = data['GPS_t'].to_numpy()[:GPS_length]
gnss_t = gnss_t.squeeze()

gnss_start_index = 105
gnss_end_index = 235
# acc = np.linalg.norm(imu_f['acc'], axis=0)
distance = np.linalg.norm(gnss['gnss'], axis=0)

imu_index = []
for k in (gnss_start_index, gnss_end_index):
    t = gnss_t[k]
    for i in range(len(imu_t)):
        if abs(imu_t[i] - t) < 0.01:
            imu_index.append(i)
            break

static_imu_data = np.concatenate((imu_f["acc"], imu_w["ang_vel"])).T
static_imu_data = static_imu_data[imu_index[0]:imu_index[1], :]
df = pd.DataFrame(static_imu_data, columns=[
    "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"])

if save == True:
    df.to_csv("static_imu_data.csv", index=False)

# ellipsoid calibration
acc_offset = np.array([[0.76056858, 0.35444493, 9.58094738 - 9.81]])
acc_gain = np.array([[7.51677859e-01, -3.62905694e-03, 3.11399172e-04],
                     [-6.17840586e-03, 4.56642887e-01, 1.63579906e-01],
                     [2.69327736e-04, 8.31018959e-02, 8.98808983e-01]])
result = acc_gain.dot((imu_f['acc'].T - acc_offset).T)

noise = np.random.normal(0,0.1,100)
print(noise)
