import argparse

import matplotlib.pyplot as plt
from apple_pick_annotater.apple_annotater import AppleAnnotater
from apple_pick_annotater.apple_pick import ApplePick

DATA_FOLDER_PATH = "F:/Orchard Dataset/"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Annotate apple pick data")
    parser.add_argument(
        "-p",
        "--plot_function",
        type=str,
        default="force",
        choices=("force", "plus"),
        help="Plotting function to use; force or plus (force plus acceleration)",
    )
    parser.add_argument(
        "-c",
        "--channels_to_record",
        type=str,
        nargs="+",
        default="a",
        help="Recording channels",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="F:/Orchard Dataset/",
        help="Path to the folder containing the data",
    )
    return parser.parse_args()


def plot_event_channels(pick: ApplePick):
    """
    This is an example plotting function for data annotation.

    The plotting function is what determines what data you see while
    annotating, however you are still free to choose completely unrelated data
    channels for export.
    """

    # trial.remove_approach()

    f = pick.get_force_mag()
    ax1 = plt.subplot(4, 1, 1, picker=5)
    ax1.plot(pick.t, f, picker=True)

    try:
        a1, a2, a3 = pick.get_accel_mag()

        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(pick.t, a1)
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(pick.t, a2)
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(pick.t, a3)
    except:
        print("no IMU  data")

    plt.gcf().suptitle(
        f"{pick.orchard_name}, {pick.trial_name}: "
        f"{pick.metadata.Outcome[0]} {pick.metadata.Controller[0]}"
    )


def plot_force(pick: ApplePick):
    """
    This is an example plotting function for data annotation.

    The plotting function is what determines what data you see while
    annotating, however you are still free to choose completely unrelated data
    channels for export.
    """
    ax = plt.subplot(1, 1, 1, picker=5)
    ax.plot(pick.t, pick.get_force_mag(), ".", picker=True)
    plt.gcf().suptitle(
        f"{pick.orchard_name}, {pick.trial_name}: "
        f"{pick.metadata.Outcome[0]} {pick.metadata.Controller[0]}"
    )


if __name__ == "__main__":
    args = parse_arguments()
    input_path = args.input_path
    plot_function = plot_force if args.plot_function == "plus" else plot_event_channels
    channels = args.channels_to_record
    event_annotater = AppleAnnotater(
        input_path, plot_function, channels=channels, window=500, save=True
    )
    event_annotater.run()
