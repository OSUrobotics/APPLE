# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:24:28 2024

@author: mcrav
"""
import os
import time
from typing import Callable, Iterable, Any

from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backend_bases import PickEvent

from .apple_pick import ApplePick
from .config import configure_figure


class AppleAnnotater:

    def __init__(
        self,
        dataset_path: str,
        plot_function: Callable[[ApplePick], Any],
        orchards: list[str] = ("Envy 2D", "Gala 2D", "Gala 3D"),
        channels: list[str] = None,
        window: int = 0,
        data_loaders: int = 3,
        save: bool = False,
    ):
        """
        This class is a tool that allows you to click on points in plotted data
        from the APPLE dataset and automatically record the time that is
        associated with that data. You must specify a path to the APPLE dataset
        and a function that plots the data you wish to see. An example function
        is provided in apple_annotater.py.

        You may additionally specify a list of data channels you would like to
        be recorded alongside the times. These channels should be specified as
        a list of strings and a wide renge of strings is accepted for each
        channel. For example, the magnitude of the force on the wrist of the
        robot can be specified as "f", "f_mag", or "force". If you want to make
        sure a given string will provide the expected data, check the function
        export_data().

        This tool should be run from the folder where the exported data is to
        be stored. The times will be stored as event_times.csv.
        """
        # Create the interface
        configure_figure()
        self.show_loading()

        # Set member variables
        self.path = dataset_path
        self.output_path = os.path.join(os.getcwd(), time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.output_path, exist_ok=True)
        self.plot_function = plot_function
        self.channels = channels
        self.window = window
        self.save = save
        # Check that there is data to load
        if not isinstance(orchards, Iterable) or len(orchards) == 0:
            raise ValueError("Orchards must be a non-empty list of strings")
        self.orchards = orchards
        self.picks = self.get_picks()
        if len(self.picks) == 0:
            raise ValueError("No trials found in the specified orchards")
        self.pick_data: list[Future | None] = [None] * len(self.picks)
        self.times = [0] * len(self.picks)
        self.time_idx = [0] * len(self.picks)

        self.index = 0
        # Start loading data
        self.max_threads = data_loaders
        self.executor = ThreadPoolExecutor(max_workers=self.max_threads)
        self.load_data()
        self.wait_for_data()
        self.plot_data()
        self.schedule_data_loaders()

        # self.orchard_idx = 0
        #
        # self.pick_names = []
        # self.pick_idx = 0

        self.req_data = []

        # self.update_orchard()
        # self.update_trial()

    @staticmethod
    def show_loading():
        """Display a loading message on the plot"""
        plt.clf()
        plt.text(0.5, 0.5, "Loading data...", fontsize=24, ha="center")
        plt.gca().set_axis_off()
        plt.pause(0.1)

    def schedule_data_loaders(self):
        open_idx = -1
        while open_idx - self.index <= self.max_threads:
            try:
                open_idx = self.pick_data.index(None, self.index + 1)
            except ValueError:
                # No None's left, we're done
                return
            # No need to load if we've already loaded enough
            if open_idx - self.index <= self.max_threads:
                self.load_data(open_idx)

    def load_data(self, idx: int = None):
        if idx is None:
            idx = self.index
        self.pick_data[idx] = self.executor.submit(
            ApplePick.from_csv, self.path, self.picks[idx][0], self.picks[idx][1]
        )

    def wait_for_data(self):
        while not self.pick_data[self.index].done():
            plt.pause(0.1)
        self.current_pick = self.pick_data[self.index].result()
        self.pick_data[self.index] = None  # Clear the data

    def plot_data(self):
        self.plot_function(self.current_pick)
        plt.gcf().canvas.mpl_connect("pick_event", self.on_pick)
        plt.show() #added to solve long data loading screen

    def run(self):
        # Creates an infinite loop that allows the user to click on points in
        # the data and record the time associated with that point.
        while True:
            plt.pause(0.1)

    def on_pick(self, event: PickEvent):
        # Only respond if the line was clicked on
        if isinstance(event.artist, Line2D):
            self.show_loading()
            self.export_times(event.ind[0])
            self.export_data()
            self.index += 1
            self.wait_for_data()
            self.plot_data()
            self.schedule_data_loaders()

    def get_picks(self):
        picks = []
        for orchard in self.orchards:
            folders = os.listdir(os.path.join(self.path, orchard))
            for folder in folders:
                # Folders indicate trials
                if os.path.isdir(os.path.join(self.path, orchard, folder)):
                    picks.append((orchard, folder))
        return picks

    # def select_next(self):
    #
    #     if self.pick_idx == len(self.pick_names) - 1:
    #         print("This was the last pick in this orchard")
    #         plt.clf()
    #         # if self.fig is not None:
    #         #     plt.close(self.fig)
    #         self.pick_idx = 0
    #         self.orchard_idx += 1
    #         self.update_orchard()
    #     else:
    #         self.export_data()
    #         self.pick_idx += 1
    #
    #     print("currently on {}".format(self.pick_names[self.pick_idx]))

    # def update_orchard(self):
    #     print("Orchard is now {}".format(self.orchards[self.orchard_idx]))
    #     self.pick_names = os.listdir(self.path + self.orchards[self.orchard_idx] + "/")
    #     self.times = [0.0] * len(self.pick_names)

    # def update_trial(self):
    #     orchard = self.orchards[self.orchard_idx]
    #     pick = self.pick_names[self.pick_idx]
    #
    #     if df_metadata.size > 0:
    #         self.current_pick = ApplePick.from_csv(os.path.join(self.path, orchard, pick))
    #         self.plot_function(self.current_pick)
    #         self.fig.canvas.mpl_connect("pick_event", self.on_pick)
    #     else:
    #         print("no metadata :(")
    #         self.select_next()
    #         self.update_trial()

    def export_times(self, time_index):
        self.time_idx[self.index] = time_index
        self.times[self.index] = self.current_pick.t[time_index]
        pick_paths = [os.path.join(self.path, p[0], p[1]) for p in self.picks]
        out = pd.DataFrame({"pick": pick_paths, "times": self.times})
        file_name = os.path.join(self.output_path, "event_times.csv")
        out.to_csv(file_name)

    # def get_grid(self, channels):

    #     n = len(channels)
    #     if n%2 == 0:
    #         fig, axes = plt.subplots(n/2,2, picker=5)
    #     elif n%3 == 0:
    #         fig, axes = plt.subplots(n/3,3, picker=5)

    def export_data(self):
        """

        This function outputs a .csv file with the requested data channels from
        self.channels. A wide range of string inputs are accepted for each
        possible timeseries.

        """
        if self.channels is None:
            return

        else:
            start = max(0, self.time_idx[self.index] - self.window)
            end = min(self.time_idx[self.index] + self.window, len(self.current_pick.t))

            data = {"time": self.current_pick.t[start:end]}

            # accelerations by channel
            a_x = [
                "ax",
                "a_x",
                "accel_x",
                "acceleration_x",
                "x_accel",
                "x_acceleration",
            ]
            if any(name in self.channels for name in a_x):
                data["a_x_1"] = self.current_pick.imu1_data[0, start:end]
                data["a_x_2"] = self.current_pick.imu2_data[0, start:end]
                data["a_x_3"] = self.current_pick.imu3_data[0, start:end]

            a_y = [
                "ay",
                "a_y",
                "accel_y",
                "acceleration_y",
                "y_accel",
                "y_acceleration",
            ]
            if any(name in self.channels for name in a_y):
                data["a_y_1"] = self.current_pick.imu1_data[1, start:end]
                data["a_y_2"] = self.current_pick.imu2_data[1, start:end]
                data["a_y_3"] = self.current_pick.imu3_data[1, start:end]

            a_z = [
                "az",
                "a_z",
                "accel_z",
                "acceleration_z",
                "z_accel",
                "z_acceleration",
            ]
            if any(name in self.channels for name in a_z):
                data["a_z_1"] = self.current_pick.imu1_data[2, start:end]
                data["a_z_2"] = self.current_pick.imu2_data[2, start:end]
                data["a_z_3"] = self.current_pick.imu3_data[2, start:end]

            # gyroscopes by channel

            g_x = ["gx", "g_x", "gyro_x", "gyroscope_x", "x_gyro", "x_gyroscope"]
            if any(name in self.channels for name in g_x):
                data["g_x_1"] = self.current_pick.imu1_data[3, start:end]
                data["g_x_2"] = self.current_pick.imu2_data[3, start:end]
                data["g_x_3"] = self.current_pick.imu3_data[3, start:end]

            g_y = ["gy", "g_y", "gyro_y", "gyroscope_y", "y_gyro", "y_gyroscope"]
            if any(name in self.channels for name in g_y):
                data["g_y_1"] = self.current_pick.imu1_data[4, start:end]
                data["g_y_2"] = self.current_pick.imu2_data[4, start:end]
                data["g_y_3"] = self.current_pick.imu3_data[4, start:end]

            g_z = ["gz", "g_z", "gyro_z", "gyroscope_z", "z_gyro", "z_gyroscope"]
            if any(name in self.channels for name in g_z):
                data["g_z_1"] = self.current_pick.imu1_data[5, start:end]
                data["g_z_2"] = self.current_pick.imu2_data[5, start:end]
                data["g_z_3"] = self.current_pick.imu3_data[5, start:end]

            # net acceleration
            a = ["a", "accel", "acceleration", "a_mag"]
            if any(name in self.channels for name in a):
                try:
                    a1, a2, a3 = self.current_pick.get_accel_mag()
                    data["a_1"] = a1[start:end]
                    data["a_2"] = a2[start:end]
                    data["a_3"] = a3[start:end]
                except:
                    print("acceleration data not available")    

            # forces

            f_x = ["fx", "f_x", "force_x", "x_force"]
            if any(name in self.channels for name in f_x):
                data["f_x"] = self.current_pick.wrench_data[3, start:end]

            f_y = ["fy", "f_y", "force_y", "y_force"]
            if any(name in self.channels for name in f_y):
                data["f_y"] = self.current_pick.wrench_data[4, start:end]

            f_z = ["fz", "f_z", "force_z", "z_force"]
            if any(name in self.channels for name in f_z):
                data["f_z"] = self.current_pick.wrench_data[5, start:end]

            f = ["f", "f_mag", "force"]
            if any(name in self.channels for name in f):
                f = self.current_pick.get_force_mag()
                data["f"] = f[start:end]

            # torques
            tau_x = ["tx", "t_x", "taux", "tau_x", "torque_x", "x_torque"]
            if any(name in self.channels for name in tau_x):
                data["tau_x"] = self.current_pick.wrench_data[0, start:end]

            tau_y = ["ty", "t_y", "tauy", "tau_y", "torque_y", "y_torque"]
            if any(name in self.channels for name in tau_y):
                data["tau_y"] = self.current_pick.wrench_data[1, start:end]

            tau_z = ["tz", "t_z", "tauz", "tau_z", "torque_z", "z_torque"]
            if any(name in self.channels for name in tau_z):
                data["tau_x"] = self.current_pick.wrench_data[2, start:end]

            # export
            self.req_data.append(data)

            if self.save:
                out = pd.DataFrame.from_dict(data)
                file_name = os.path.join(
                    self.output_path,
                    self.current_pick.orchard_name + 
                    "_" + self.picks[self.index][1] + "_data_req.csv",
                )
                out.to_csv(file_name)
