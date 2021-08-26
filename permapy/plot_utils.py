
def plot_raster(raster,output_path,title,x_label,y_label,style):
    """
    Function to plot and save raster as a png image
    """
    array= raster.read(1)
    print("Array: \n",array)
    plt.imshow(array, cmap=style)
    plt.title(title,fontsize=20)
    plt.ylabel(y_label,fontsize=18)
    plt.xlabel(x_label,fontsize=18)
    plt.savefig(f"{output_path}.png")
    plt.show()

def saving_input_vars_histograms(df,output_path:str,suptitle:str):
    """
    Function to generate and sabe histograms
    """
    df.hist(layout=(10,4),figsize=(20,20))
    plt.suptitle(suptitle,fontsize = 40)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path)
    plt.show()
    plt.clf()