import xarray as xr
import numpy as np
from typing import Union, Dict

class ERA5DataProcessor:
    """ This Class holds all the necessary steps to process and treat a dataset and set it up for plot. 
    """
    def __init__(self, ds: xr.Dataset, variables: Union[list,dict], date_range: Union[list,tuple],
                 spatial_range:dict,
                 convert_longitude: bool = True, include_attributes: bool = True):
        """
        ds : xarray.Dataset
            The source dataset containing meteorological variables and coordinates
        variables : dict or list
            Dictionary of data variables to include or list of variable names.
            If dict, keys are variable names in the source dataset and values are new names.
            If list, the same variable names will be used in the output.
        date_range : list or tuple
            Date range for filtering the dataset [start_date, end_date] (format: ["YYYY-MM-DD", "YYYY-MM-DD"])
        spatial_range : dict
            Dictionary defining the spatial boundaries with keys:
            - 'lat': [min_lat, max_lat]
            - 'lon': [min_lon, max_lon]
        convert_longitude : bool, optional (default=True)
            Whether to convert longitude from 0-360 to -180 to 180
        include_wind_speed : bool, optional (default=True)
            Whether to calculate and include wind_speed in the output dataset
        include_attributes : bool, optional (default=True)
            Whether to copy variable attributes from the source dataset
        """
        self.dataset = ds.copy() #original dataset requisition 
        self.loaded_dataset = None  # hold the processed dataset. This has data loaded. 
        self.variables = variables
        self.date_range = date_range
        self.spatial_range = spatial_range
        self.convert_longitude = convert_longitude
        self.include_attributes = include_attributes

        
    def process_data(self):
        """Filter by date, select variables, convert/sort longitude, and crop by spatial bounds."""
        start_date, end_date = self.date_range
        latitude_range = self.spatial_range['lat']
        longitude_range = self.spatial_range['lon']
    
        # Filter date
        data = self.dataset.sel(valid_time=slice(start_date, end_date))
        
        # Select variables
        var_list = self.variables if isinstance(self.variables, list) else list(self.variables.keys())
        data = data[var_list]
        
        # Convert longitude if needed
        if self.convert_longitude:
            data = data.assign_coords(longitude=(data.longitude + 180) % 360 - 180)
            data = data.sortby('longitude')
        
        # Select region of interest
        data = data.sel(
            latitude=slice(latitude_range[1], latitude_range[0]),
            longitude=slice(longitude_range[0], longitude_range[1])
        ).load()
        
        self.loaded_dataset = data

    def extract_components_by_given_timestep(self,
                                             timestep:int,
                                             extract_variables: Union[list,dict],
                                             lat_long: bool = True) -> Dict[str, xr.Dataset]:
        """Extract a specific timestep for all selected variables.
        Return a dict containing all variables.

         timestep : int or None
            The time index to extract a especific timestep after the completion of date selection.  """
        
        if extract_variables is None: 
            extract_variables = ValueError('Please select variables to be extracted.') 
            
        if self.loaded_dataset is None:
            raise ValueError('You must run process_data() before extracting timestep.')
    
        data = self.loaded_dataset
    
        if timestep >= data.valid_time.size:
            raise IndexError(f"Timestep {timestep} is out of bounds for the selected data.")

        # transform in a list
        var_list = extract_variables if isinstance(extract_variables, list) else list(extract_variables.keys())
    
        output_vars = {}
        for var in var_list:
            if var in list(data.data_vars):
                output_vars[var] = data[var].isel(valid_time=timestep)
            else:
                raise KeyError(f"Variable {var} not found in dataset.")

        if lat_long:
            output_vars['lat'] = data['latitude']
            output_vars['long'] = data['longitude']
            
        return output_vars

    def calculate_wind_speed(self, u_component: str = 'u10', v_component: str = 'v10', new_var_name: str = 'wind_speed'):
        """
        Calculate wind speed from u and v components and add it to the processed dataset.
        
        Parameters:
        - u_component (str): Name of the east-west wind component.
        - v_component (str): Name of the north-south wind component.
        - new_var_name (str): Name of the new wind speed variable.
        """
        if self.loaded_dataset is None:
            raise ValueError('No processed data available. Run process_data() first.')

        ds = self.loaded_dataset

        if u_component not in ds or v_component not in ds:
            raise ValueError(f"Variables {u_component} and/or {v_component} not found in processed dataset.")

        wind_speed = np.sqrt(ds[u_component] ** 2 + ds[v_component] ** 2)
        ds[new_var_name] = wind_speed
        
        if self.include_attributes:
            ds[new_var_name].attrs = {
                'long_name': f'Wind Speed calculated from {u_component} and {v_component}',
                'units': ds[u_component].attrs.get('units', 'm s**-1')
                if hasattr(ds[u_component], 'attrs') and 'units' in ds[u_component].attrs
                else 'm s**-1'
            }

        self.loaded_dataset = ds

    def subsample_data(self, step: int = 2) -> xr.Dataset:
        """
        Subsample the processed dataset by a given step size.
        
        Returns a new xarray.Dataset.
        """
        if self.loaded_dataset is None:
            raise ValueError('You must run process_data() before subsampling.')

        ds = self.loaded_dataset

        # Subsample coordinates
        lon_sub = ds['longitude'][::step]
        lat_sub = ds['latitude'][::step]
        
        # Subsample available variables
        subsampled_vars = {}
        for var in ds.data_vars:
            subsampled_vars[var] = ds[var].isel(latitude=slice(None, None, step),
                                                longitude=slice(None, None, step))
        
        subsampled_ds = xr.Dataset(subsampled_vars)
        
        # Reassign coordinates
        subsampled_ds = subsampled_ds.assign_coords({
            'longitude': lon_sub,
            'latitude': lat_sub
        })

        return subsampled_ds

    def get_processed_data(self) -> xr.Dataset:
        """
        Returns the processed dataset.
        """
        if self.loaded_dataset is None:
            raise ValueError('No processed data available. Run process_data() first.')
        
        return self.loaded_dataset
