from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class SpeciesInRasterPlotter:
    def __init__(self) -> None:
        pass

    @classmethod
    def plot_all_points(cls, raster_array, ix, iy):
        plt.imshow(raster_array, cmap="terrain")
        plt.scatter(x=ix, y=raster_array.shape[1] - iy, c="r", s=10)
        plt.savefig("species_in_raster.png")
        plt.clf()

    @classmethod
    def plot_one_point(cls, raster_array, x: float, y: float):

        plt.imshow(raster_array, cmap="terrain")
        plt.scatter(x=x, y=raster_array.shape[1] - y, c="r", s=10)
        plt.savefig("species_point_in_raster.png")
        plt.close()
        plt.clf()


class RasterPlotter:
    def __init__(self) -> None:
        pass

    def __get_map_plot_cmap(self, raster_array):

        raster_array[raster_array == -9999.0] = -0.001
        norm = matplotlib.colors.Normalize(-0.001, 100)
        colors = [
            [norm(-0.001), "black"],
            [norm(0), "0.95"],
            [norm(10), "sienna"],
            [norm(30), "wheat"],
            [norm(50), "cornsilk"],
            [norm(80), "yellowgreen"],
            [norm(100), "green"],
        ]
        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        custom_cmap.set_bad(color="black")

        return custom_cmap

    def __get_rsep_map(self, raster_array):
        raster_array[raster_array == -9999.0] = 0
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [
            [norm(-1), "red"],
            [norm(0), "black"],
            [norm(1), "blue"],
        ]
        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        return custom_cmap

    def plot(self, raster_array, save_path: Path, cmap_style: str):
        # inferno = cm.get_cmap('inferno', 1)
        # terrain = cm.get_cmap('terrain', 10)# combine it all
        # newcolors = np.vstack((inferno(np.linspace(0, 1, 1)),
        #                terrain(np.linspace(0, 1, 10))))
        # mycm = ListedColormap(newcolors, name='mycm')
        import pdb

        pdb.set_trace()
        custom_cmap = (
            self.__get_map_plot_cmap(raster_array)
            if cmap_style == "normal_map"
            else self.__get_rsep_map(raster_array)
        )
        plt.plot(raster_array, cmap=custom_cmap)
        plt.savefig(save_path)
        plt.clf()
