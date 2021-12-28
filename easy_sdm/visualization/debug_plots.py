import matplotlib.pyplot as plt

class SpeciesInRasterPlotter():
    def __init__(self) -> None:
        pass

    @classmethod
    def plot(cls, raster_array,ix,iy):

        plt.imshow(raster_array, cmap='terrain')
        plt.scatter(x=ix, y=raster_array.shape[1]-iy, c='r', s=10)
        plt.savefig('species_in_raster.png')

class RasterPlotter():
    def __init__(self) -> None:
        pass

    @classmethod
    def plot(cls, raster_array):

        plt.imshow(raster_array, cmap='terrain')
        plt.savefig('raster.png')