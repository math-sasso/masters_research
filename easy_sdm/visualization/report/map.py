import os
import matplotlib.pyplot as plt

class MapPlotter(object):
    def __init__(self) -> None:
        pass

    def saving_kfold_result_maps(
        species_name,
        species_kfold_preditions_folder,
        utils_methods,
        raster_utils,
        output_folder,
        country_mask_reference,
        n_levels,
    ):

        # Getting Lad Reference Infos
        land_reference_array, _, xgrid, ygrid, _, _ = raster_utils.get_raster_infos(
            country_mask_reference
        )

        # Getting Meshgrids
        # X, Y = np.meshgrid(xgrid, ygrid[::-1])

        X, Y = xgrid, ygrid[::-1]

        kfold_dirs = [
            os.path.join(species_kfold_preditions_folder, name)
            for name in os.listdir(species_kfold_preditions_folder)
            if os.path.isdir(os.path.join(species_kfold_preditions_folder, name))
        ]
        for i, fold in enumerate(kfold_dirs):
            plt.figure(figsize=(8, 8))
            Z = utils_methods.retrieve_data_from_np_array(
                os.path.join(fold, "Land_Prediction.npy")
            )
            coords_train = utils_methods.retrieve_data_from_np_array(
                os.path.join(fold, "Coords_Train.npy")
            )
            coords_test = utils_methods.retrieve_data_from_np_array(
                os.path.join(fold, "Coords_Test.npy")
            )

            # Setting titles and labels
            plt.title(
                f"Prediction k-fold{i+1} \n for species {species_name}",
                fontsize=16,
                fontname="Arial",
            )
            plt.ylabel("Latitude[degrees]", fontsize=14, fontname="Arial")
            plt.xlabel("Longitude[degrees]", fontsize=14, fontname="Arial")

            # Plot country map
            plt.contour(
                X, Y, land_reference_array, levels=[10], colors="k", linestyles="solid"
            )

            # Creating personalized cmap
            positive_blue_colors = plt.cm.Blues(np.linspace(0, 1, n_levels))
            negative_red_colors = plt.cm.Reds(np.ones(n_levels))
            colors = np.vstack((negative_red_colors, positive_blue_colors))
            mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)
            levels = np.linspace(-1, 1, n_levels)

            # print('levels: ',levels)
            plt.contourf(X, Y, Z, levels=levels, cmap=mymap, vmin=-1, vmax=1)
            plt.colorbar(format="%.2f")

            # Plot points on map
            plt.scatter(
                coords_train[:, 1],
                coords_train[:, 0],
                s=2 ** 3,
                c="green",
                marker="^",
                label="train",
            )
            plt.scatter(
                coords_test[:, 1],
                coords_test[:, 0],
                s=3 ** 2,
                c="k",
                marker="x",
                label="test",
            )

            # Saving results
            plt.legend(loc="upper right")
            utils_methods.create_folder_structure(output_folder)
            plt.savefig(f"{output_folder}/land_map_prediction_kfold{i+1}.png")
            plt.show()
            plt.clf()

    def plot_raster(raster, output_path, title, x_label, y_label, style):
        """
        Function to plot and save raster as a png image
        """
        array = raster.read(1)
        print("Array: \n", array)
        plt.imshow(array, cmap=style)
        plt.title(title, fontsize=20)
        plt.ylabel(y_label, fontsize=18)
        plt.xlabel(x_label, fontsize=18)
        plt.savefig(f"{output_path}.png")
        plt.show()

    def plot_occurrences_on_map(
        specie_gdf,
        result_gibf_queries_maps_root_folder,
        species_name,
        utils_methods,
        brazil,
    ):
        species_gibf_queries_maps_root_folder = os.path.join(
            result_gibf_queries_maps_root_folder, species_name
        )
        # create folder structure if necessary
        utils_methods.create_folder_structure(species_gibf_queries_maps_root_folder)

        # Create a figure with one subplot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot polygons
        brazil.get_country_level_gdf().plot(ax=ax, facecolor="gray")
        # Plot points
        specie_gdf.plot(ax=ax, color="blue", markersize=5)

        plt.title(
            f"{species_name} occurrences on \n Brazilian Territory",
            fontsize=16,
            fontname="Arial",
        )
        plt.ylabel("Latitude [degrees]", fontsize=14, fontname="Arial")
        plt.xlabel("Longitude [degrees]", fontsize=14, fontname="Arial")

        plt.tight_layout()
        plt.savefig(f"{species_gibf_queries_maps_root_folder}/{species_name}.png")
        plt.show()
        plt.clf()
