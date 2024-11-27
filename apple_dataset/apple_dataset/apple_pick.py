import os
from typing import Self
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from .config import COLOR


class ApplePick:

    def __init__(self, orchard_name, trial_name, data, metadata):
        self.orchard_name = orchard_name
        self.trial_name = trial_name
        self.data = data.dropna()
        self.metadata = metadata.dropna()
        try:
            self.b1, self.b2, self.b3 = self.get_branch_coords()
            self.joint_loc = self.get_joint_location()
        except:
            print("Invalid metadata :(")

        self.t = self.data.Time.to_list() - self.data.Time[self.data.Time.index[0]]

        if "Module1Accel_1" in self.data:
            self.imu1_data, self.imu2_data, self.imu3_data = self.get_imu_data()

        self.wrench_data = self.get_wrench_data()
        self.path_data = self.get_wrist_path()

    def __len__(self):
        return len(self.metadata)

    @classmethod
    def from_csv(cls, data_path, orchard_name, trial_name) -> Self:
        trial_path = os.path.join(data_path, orchard_name, trial_name)
        data = pd.read_csv(os.path.join(trial_path, "AllInterpolatedData.csv"))
        metadata = pd.read_csv(os.path.join(trial_path, "metadata.csv"))
        return cls(orchard_name, trial_name, data, metadata)

    def get_branch_coords(self):
        p1 = [
            self.metadata.IMU1Location_1[0],
            self.metadata.IMU1Location_2[0],
            self.metadata.IMU1Location_3[0],
        ]
        p2 = [
            self.metadata.IMU2Location_1[0],
            self.metadata.IMU2Location_2[0],
            self.metadata.IMU2Location_3[0],
        ]
        p3 = [
            self.metadata.IMU3Location_1[0],
            self.metadata.IMU3Location_2[0],
            self.metadata.IMU3Location_3[0],
        ]

        return p1, p2, p3

    # this function isn't possible yet because orientations are not included
    # def get_branch_rotations(self,idx):

    #     eul1 = [self.data.IMU1Orientation_1[0],
    #             self.data.IMU1Orientation_2[0],
    #             self.data.IMU1Orientation_3[0]]

    #     return eul1

    def get_joint_location(self):

        absc_jnt = [
            self.metadata.AbscissionJointLocation_1[0],
            self.metadata.AbscissionJointLocation_2[0],
            self.metadata.AbscissionJointLocation_3[0],
        ]

        return absc_jnt

    def get_wrist_transform(self, idx):

        idx = self.data.index[idx]

        transform = np.identity(4)

        q = [
            self.data.WristOrientation_1[idx],
            self.data.WristOrientation_2[idx],
            self.data.WristOrientation_3[idx],
            self.data.WristOrientation_4[idx],
        ]
        try:
            r = Rotation.from_quat(q)
        except ValueError:
            print("hey")
        R = r.as_matrix()
        transform[0:3, 0:3] = R

        transform[0, 3] = self.data.WristPosition_1[idx]
        transform[1, 3] = self.data.WristPosition_2[idx]
        transform[2, 3] = self.data.WristPosition_3[idx]

        return transform

    def get_axis_limits(self):

        b1, b2, b3 = self.get_branch_coords()
        branch_points = np.array([b1, b2, b3])
        wrist_points = np.array(
            [
                self.data.WristPosition_1,
                self.data.WristPosition_2,
                self.data.WristPosition_3,
            ]
        ).transpose()
        all_points = np.vstack([branch_points, wrist_points])

        x_min = np.min(all_points[:, 0]) - 10
        x_max = np.max(all_points[:, 0]) + 10
        y_min = np.min(all_points[:, 1]) - 10
        y_max = np.max(all_points[:, 1]) + 10
        z_min = np.min(all_points[:, 2]) - 10
        z_max = np.max(all_points[:, 2]) + 10

        x_lims = [x_min, x_max]
        y_lims = [y_min, y_max]
        z_lims = [z_min, z_max]

        return x_lims, y_lims, z_lims

    def get_imu_data(self):

        imu1 = np.array(
            [
                self.data.Module1Accel_1,
                self.data.Module1Accel_2,
                self.data.Module1Accel_3,
                self.data.Module1Gyro_1,
                self.data.Module1Gyro_2,
                self.data.Module1Gyro_3,
            ]
        ).transpose()

        imu2 = np.array(
            [
                self.data.Module2Accel_1,
                self.data.Module2Accel_2,
                self.data.Module2Accel_3,
                self.data.Module2Gyro_1,
                self.data.Module2Gyro_2,
                self.data.Module2Gyro_3,
            ]
        ).transpose()

        imu3 = np.array(
            [
                self.data.Module3Accel_1,
                self.data.Module3Accel_2,
                self.data.Module3Accel_3,
                self.data.Module3Gyro_1,
                self.data.Module3Gyro_2,
                self.data.Module3Gyro_3,
            ]
        ).transpose()

        return imu1, imu2, imu3

    def get_wrench_data(self):

        wrench_data = np.array(
            [
                self.data.WristTorque_1,
                self.data.WristTorque_2,
                self.data.WristTorque_3,
                self.data.WristForce_1,
                self.data.WristForce_2,
                self.data.WristForce_3,
            ]
        ).transpose()

        return wrench_data

    def plot_6d_data(self, ax, x, y):

        ax.plot(x, y[:, 0], color=COLOR["dark_blue"])
        ax.plot(x, y[:, 1], color=COLOR["light_blue"])
        ax.plot(x, y[:, 2], color=COLOR["teal"])
        ax.plot(x, y[:, 3], color=COLOR["dark_purple"])
        ax.plot(x, y[:, 4], color=COLOR["light_purple"])
        ax.plot(x, y[:, 5], color=COLOR["salmon"])

    def draw_transform(self, ax, T):

        px = T[0, 3]
        py = T[1, 3]
        pz = T[2, 3]

        x = T[:, 0]
        y = T[:, 1]
        z = T[:, 2]

        ax.quiver(
            px, py, pz, x[0], x[1], x[2], length=10, color=COLOR["teal"], linewidths=4
        )
        ax.quiver(
            px, py, pz, y[0], y[1], y[2], length=10, color=COLOR["teal"], linewidths=4
        )
        ax.quiver(
            px, py, pz, z[0], z[1], z[2], length=10, color=COLOR["teal"], linewidths=4
        )

        return

    def rotate_surface(self, x_grid, y_grid, z_grid, R, n):

        new_x = np.zeros(
            n**2,
        )
        new_y = np.zeros(
            n**2,
        )
        new_z = np.zeros(
            n**2,
        )

        x = x_grid.flatten()
        y = y_grid.flatten()
        z = z_grid.flatten()

        for i in range(n**2):

            point = np.array([[x[i]], [y[i]], [z[i]]])
            new_point = np.matmul(R, point)

            new_x[i] = new_point[0]
            new_y[i] = new_point[1]
            new_z[i] = new_point[2]

        new_xgrid = new_x.reshape(n, n)
        new_ygrid = new_y.reshape(n, n)
        new_zgrid = new_z.reshape(n, n)

        return new_xgrid, new_ygrid, new_zgrid

    def draw_osu_gripper(self, ax, T):

        l = 16.7  # all units are cm
        r = 5

        xc, yc, zc = self.data_for_cylinder(T, 50, r, l)

        self.draw_transform(ax, T)
        ax.plot_surface(xc, yc, zc, color=COLOR["light_blue"], alpha=0.5)

    def data_for_cylinder(self, T, n, radius, height):

        z = np.linspace(0, height, n)
        theta = np.linspace(0, 2 * np.pi, n)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        R = T[0:3, 0:3]
        new_xgrid, new_ygrid, new_zgrid = self.rotate_surface(
            x_grid, y_grid, z_grid, R, n
        )

        new_xgrid = new_xgrid + T[0, 3]
        new_ygrid = new_ygrid + T[1, 3]
        new_zgrid = new_zgrid + T[2, 3]

        return new_xgrid, new_ygrid, new_zgrid

    def data_for_elipse(self, rx, ry, rz, ax, n):

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, np.pi, n)

        # Cartesian coordinates that correspond to the spherical angles:
        # (this is the equation of an ellipsoid):
        x = rx * np.outer(np.cos(u), np.sin(v))
        y = ry * np.outer(np.sin(u), np.sin(v))
        z = rz * np.outer(np.ones_like(u), np.cos(v))

        return x, y, z

    def draw_apple(self, r, h, joint_loc, ax, n):
        x_local, y_local, z_local = self.data_for_elipse(r, r, h, ax, n)
        x = x_local + joint_loc[0]
        y = y_local + joint_loc[1]
        z = z_local + joint_loc[2] - h / 2
        ax.plot_surface(x, y, z, color=COLOR["salmon"])

    def draw_branch(self, p1, p2, p3, r1, r2, ax):

        x = [p1[0], p2[0], p3[0]]
        y = [p1[1], p2[1], p3[1]]
        z = [p1[2], p2[2], p3[2]]

        ax.plot(x, y, z, color=COLOR["dark_purple"], linewidth=12)

    def point2segment(self, segment_start, segment_end, point):

        p1 = np.array(segment_start)
        p2 = np.array(segment_end)
        p3 = np.array(point)

        # d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        d = np.linalg.norm(np.cross(p3 - p1, p3 - p2)) / np.linalg.norm(p2 - p1)

        return d

    def dist2lim(self):

        dist1 = self.point2segment(self.b1, self.b2, self.joint_loc)
        dist2 = self.point2segment(self.b2, self.b3, self.joint_loc)

        return min(dist1, dist2)

    def get_wrist_path(self):

        n = len(self.t)

        path = np.zeros([n, 3])

        for i in range(n):

            T = self.get_wrist_transform(i)
            path[i, :] = T[0:3, 3]

        return path

    def path_length(self):

        movement = np.linalg.norm(self.path_data - self.path_data[0, :], axis=1)
        path_length = np.sum(np.diff(movement, axis=0), axis=0)

        return path_length

    def remove_approach(self):

        downsample = 100

        movement = np.linalg.norm(self.path_data - self.path_data[0, :], axis=1)
        movement_downsample = movement[0::downsample]

        speed = np.diff(movement_downsample, axis=0)

        condition = abs(speed) < 0.01

        transitions = np.where(np.diff(condition) != 0)[0] + 1

        if len(transitions) > 4:

            start_idx = transitions[3] * downsample

            self.path_data = self.path_data[start_idx:, :]
            self.wrench_data = self.wrench_data[start_idx:, :]
            self.t = self.t[start_idx:]

            try:
                self.imu1_data = self.imu1_data[start_idx:, :]
                self.imu2_data = self.imu2_data[start_idx:, :]
                self.imu3_data = self.imu3_data[start_idx:, :]
            except:
                pass

    def get_peak_force(self):
        f = self.wrench_data[:, 3:6]
        f_norm = np.linalg.norm(f, axis=1)

        return np.max(f_norm)

    def get_force_mag(self):
        f = self.wrench_data[:, 3:6]
        f_norm = np.linalg.norm(f, axis=1)

        return f_norm

    def get_accel_mag(self):
        a1 = self.imu1_data[:, 0:3]
        a1_norm = np.linalg.norm(a1, axis=1)
        a2 = self.imu2_data[:, 0:3]
        a2_norm = np.linalg.norm(a2, axis=1)
        a3 = self.imu3_data[:, 0:3]
        a3_norm = np.linalg.norm(a3, axis=1)

        return a1_norm, a2_norm, a3_norm
