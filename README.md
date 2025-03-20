# Climate Data Visualization: Wind and Temperature Animation

This repository contains a Jupyter notebook that demonstrates techniques for visualizing climate data (ScenarioMIP SSP3-7.0 scenario ERA5). The notebook creates field vector animation and temperature distributions over Europe.

## Overview

This project focuses on:
1. Accessing and processing climate data from the DestinE Earth Data Hub
2. Creating static visualizations of wind vector fields and temperature distributions
3. Generating animations of flowing wind patterns overlaid on temperature maps
4. Applying advanced techniques for streamline visualization with dynamic transparency

## Features

- **Data Access**: Connects to the DestinE Earth Data Hub to retrieve IFS-NEMO high-resolution climate data
- **Geographic Filtering**: Extracts data for the European domain (35°N-71°N, 10°W-40°E)
- **Visualization Techniques**:
  - Wind vector field visualization using streamlines
  - Temperature distribution display with proper color mapping
  - Animated flow visualization with dynamic transparency
- **Animation Generation**: Creates MP4 animations of wind patterns and temperature data

## Dependencies

- xarray
- matplotlib
- cartopy
- numpy
- tqdm
- loguru

## Getting Started

1. Clone this repository:
```bash
git clone git@github.com:sebastien-tetaud/lombric-quiver.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Source

The notebook uses data from the DestinE Earth Data Hub, specifically:
- Dataset: ScenarioMIP-SSP3-7.0-IFS-NEMO-0001-high-sfc-v0.zarr
- Variables:
  - t2m: 2-meter temperature (K)
  - u10: 10-meter zonal wind component (m/s)
  - v10: 10-meter meridional wind component (m/s)

## Example Output

The final output is an MP4 animation showing temperature as a color-coded map with animated wind streamlines flowing across Europe.

<video src="assets/era5_europe_t2m_wind.mp4" width="320" height="240" controls></video>


## Customization

The notebook includes several parameters that can be adjusted:
- Geographic region (latitude_range, longitude_range)
- Time period (start_date, end_date)
- Animation properties (frames, fps)
- Visualization settings (colormap, density of streamlines)

## License

Apache License 2.0

## Acknowledgments

- [DestinE Earth Data Hub](https://earthdatahub.destine.eu/)

## Contact

- [sebastien.tetaud@esa.int](sebastien.tetaud@esa.int)
- [tetaud.sebastien@gmail.com](tetaud.sebastien@gmail.com)

