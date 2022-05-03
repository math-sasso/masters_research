import matplotlib.pyplot as plt


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

    @classmethod
    def plot(cls, raster_array):
        import numpy as np
        import matplotlib
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

        # inferno = cm.get_cmap('inferno', 1)
        # terrain = cm.get_cmap('terrain', 10)# combine it all
        # newcolors = np.vstack((inferno(np.linspace(0, 1, 1)),
        #                terrain(np.linspace(0, 1, 10))))
        # mycm = ListedColormap(newcolors, name='mycm')
        plt.imshow(raster_array, cmap=custom_cmap)
        plt.show()
        plt.clf()
