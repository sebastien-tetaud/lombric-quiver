import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature  
import numpy as np
import xarray as xr


def create_layer_base(dict_extract_var, var_heatmap = 'wind_speed', dpi=100, height_inches=9,
                      crs_transform=ccrs.ccrs.PlateCarree(), cmap='plasma_r', 
                      stream_lines=False, dataset_subsampled=None, **kwargs_streamplot):
    """
    Create a base layer for plotting, the base layer includes a heatmap on wind speed and optionally streamlines.
    Save as a png file with specified dimensions.
    
    Parameters:
    dict_extract_var (dict): Dictionary containing 'long', 'lat', and 'wind_speed' keys.
    var_heatmap (str): The variable to be used for the heatmap. By default, it is set to 'wind_speed'. It should be a key in the `dict_extract_var` dictionary.
    cmap (str): Colormap for the heatmap. By default, it is set to 'plasma_r'. Refers to https://matplotlib.org/stable/gallery/color/colormap_reference.html 
    crs_transform (cartopy.crs.Projection): The coordinate reference system for the plot. By default, it is set to PlateCarree. It can be any valid Cartopy projection, such as `ccrs.Mercator()`, `ccrs.LambertConformal()`, etc.
    dpi (int): Dots per inch for the figure resolution. By default, it is set to 100.
    height_inches (int): Height of the figure in inches. By default, it is set to 9 inches. This height is highly related with Manim canvas.
    stream_lines (bool): Whether to include streamlines in the plot.   By default, it is set to False.
    dataset_subsampled (xr.dataset): xarray dataset containing subsampled data for streamlines. Only required if `stream_lines` is True.
    kwargs_streamplot: Additional keyword arguments for the streamplot function, such as `density`, `linewidth`, etc.
    
    Returns: Save the fig as a png file with the name "base_layer.png" with exact dimensions.
    
    """
    # Check if required keys are in the dictionary
    if not all(key in dict_extract_var for key in ['long', 'lat']):
        raise ValueError("Input dictionary must contain 'long', 'lat'.")
    if not all(key in dict_extract_var for key in [var_heatmap]):
        raise ValueError(f"Input dictionary must contain '{var_heatmap}'.")
    
    
    # Get the geographic extent of your data
    lon_min = dict_extract_var['long'].min()
    lon_max = dict_extract_var['long'].max()
    lat_min = dict_extract_var['lat'].min()
    lat_max = dict_extract_var['lat'].max()

    # Calculate the aspect ratio (width/height) of your geographic area
    # Account for latitude distortion with cosine correction
    avg_latitude = (lat_max + lat_min) / 2
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    # Aspect ratio calculation
    aspect_ratio = (lon_range * np.cos(np.radians(avg_latitude))) / lat_range

    # Set fixed height in inches and calculate width
    height_inches = height_inches
    width_inches = height_inches * (1/aspect_ratio)

    print(f"Geographic extent: {lon_min:.2f}°E to {lon_max:.2f}°E, {lat_min:.2f}°N to {lat_max:.2f}°N")
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    print(f"Figure dimensions: {width_inches:.2f} × {height_inches:.2f} inches")

    # Create figure with calculated dimensions
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_inches, height_inches),
                        dpi=dpi,
                        subplot_kw={'projection': crs_transform})

    # Set the geographic extent to match your data
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs_transform)

    # Add features to the map 
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND)

    # Create a heatmap based on the velocity 
    heatmap = ax.pcolormesh(dict_extract_var['long'],
                            dict_extract_var['lat'],
                            dict_extract_var[var_heatmap],
                            cmap=cmap,
                            transform=crs_transform)

    if stream_lines:
        if dataset_subsampled is None:
            raise ValueError("dataset_subsampled must be provided when stream_lines is True.")
        if not isinstance(dataset_subsampled, xr.Dataset):
            raise ValueError("dataset_subsampled must be an xarray Dataset when stream_lines is True.")
        
        # Ensure the dataset has the required variables
        required_vars = ['longitude', 'latitude', 'u10', 'v10'] 
        if not all(var in dataset_subsampled for var in required_vars):
            raise ValueError(f"dataset_subsampled must contain the following variables: {required_vars}")
        
        
        # Initialize the streamplot
        stream = ax.streamplot(dataset_subsampled['longitude'].values,
                                dataset_subsampled['latitude'].values,
                                dataset_subsampled['u10'],
                                dataset_subsampled['v10'],
                                color="white",
                                density=10,
                                linewidth=0.4,
                                **kwargs_streamplot)

    # optimize spacing
    plt.tight_layout()

    # Save to file with exact dimensions
    fig.savefig("base_layer.png", 
                dpi=dpi,
                bbox_inches='tight',
                pad_inches='layout')

    # Print final pixel dimensions
    print(f"Final image size: {int(width_inches * dpi)} × {int(height_inches * dpi)} pixels")

    plt.close(fig)