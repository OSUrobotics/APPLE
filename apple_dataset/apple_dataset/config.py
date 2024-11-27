import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use("Qt5Agg")
matplotlib.use("tkagg")

COLOR = {
    "dark_blue": [51 / 255, 34 / 255, 136 / 255],
    "dark_green": [17 / 255, 119 / 255, 51 / 255],
    "teal": [68 / 255, 170 / 255, 153 / 255],
    "light_blue": [136 / 255, 204 / 255, 238 / 255],
    "yellow": [221 / 255, 204 / 255, 119 / 255],
    "salmon": [204 / 255, 102 / 255, 119 / 255],
    "light_purple": [170 / 255, 68 / 255, 153 / 255],
    "dark_purple": [136 / 255, 34 / 255, 85 / 255],
}

FONT_SIZE = {
    "S": 12,
    "M": 16,
    "L": 18,
}


def configure_figure():
    px = 1 / plt.rcParams["figure.dpi"]  # pixels-to-inches conversion
    screen_y = plt.get_current_fig_manager().window.winfo_screenheight() * px
    # Set to a 90% of screen height, 16:9 aspect ratio
    plt.close("all")
    # Createa figure in the top-left corner

    plt.figure(figsize=(screen_y * 0.8 * 4 / 3, screen_y * 0.8))
    plt.ion()

    plt.rcParams["font.family"] = "Arial"
    plt.rc("font", size=FONT_SIZE["S"])  # controls default text sizes
    plt.rc("axes", titlesize=FONT_SIZE["L"])  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE["M"])  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=FONT_SIZE["S"])  # fontsize of the tick labels
    plt.rc("ytick", labelsize=FONT_SIZE["S"])  # fontsize of the tick labels
    plt.rc("legend", fontsize=FONT_SIZE["M"])  # legend fontsize
    plt.rc("figure", titlesize=FONT_SIZE["L"])  # fontsize of the figure title
