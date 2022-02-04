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

        plt.imshow(raster_array, cmap="terrain")
        plt.savefig("raster.png")
