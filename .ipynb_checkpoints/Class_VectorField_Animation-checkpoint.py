import numpy as np
from manim import Scene, ImageMobject, FadeIn, StreamLines, WHITE, Text
from manim import config as manim_config
from manim.utils.ipython_magic import ManimMagic
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import os
import tempfile
from IPython.display import display, HTML, Video


class VectorField_Animation:
    """
    A class that handles the creation of stream plots for wind data from ERA5 datasets
    using the Manim library, optimized for Jupyter Notebooks.
    """
    
    def __init__(self, dataset=None, background_image_path=None):
        """
        Initialize the StreamPlotHandler with a dataset and background image path.
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            The ERA5 dataset containing wind data (u10, v10)
        background_image_path : str
            Path to the background image to use for the plot
        """
        self.dataset = dataset
        self.background_image_path = background_image_path
        self.scene_class = None
        self.temp_media_dir = None
        
        # Default configuration parameters
        self.config = {
            'image_height': 7,
            'delta_y': 0.15,
            'resize_factor': 0.05,
            'stroke_width': 1.0,
            'flow_speed': 2.0,
            'time_width': 0.3,
            'animation_duration': 3,
            'dt': 0.01,
            'noise_factor': 0.0,
            'max_anchors_per_line': 100,
            'color': WHITE,
            'virtual_time': 4,
            'show_title': False,
            'title_text': "ERA5 Wind - Vector Data",
            'title_font_size': 36,
            # Jupyter-specific options
            'output_file': 'wind_stream_plot',
            'pixel_width': 854,
            'pixel_height': 480,
            'preview_frames': 30,
            'transparent': False,
            'quality': 'medium_quality',  # low_quality, medium_quality, high_quality, fourk_quality
        }
        
        # Initialize the manim config for Jupyter compatibility
        self._setup_manim_config()
    
    def _setup_manim_config(self):
        """Configure Manim for Jupyter notebook compatibility"""
        # Store the original config
        self._original_config = {
            'preview': manim_config.preview,
            'output_file': manim_config.output_file,
            'pixel_width': manim_config.pixel_width,
            'pixel_height': manim_config.pixel_height,
            'transparent': manim_config.transparent,
        }
        
        # Set temp dir for media output
        self.temp_media_dir = tempfile.mkdtemp()
        manim_config.media_dir = self.temp_media_dir
    
    def configure(self, **kwargs):
        """
        Update configuration parameters.
        
        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters to update
        
        Returns:
        --------
        self : StreamPlotHandler
            Returns self for method chaining
        """
        self.config.update(kwargs)
        
        # Update manim config based on new parameters
        if 'pixel_width' in kwargs:
            manim_config.pixel_width = self.config['pixel_width']
        if 'pixel_height' in kwargs:
            manim_config.pixel_height = self.config['pixel_height']
        if 'transparent' in kwargs:
            manim_config.transparent = self.config['transparent']
        if 'output_file' in kwargs:
            manim_config.output_file = self.config['output_file']
        if 'quality' in kwargs:
            manim_config.quality = self.config['quality']
            
        return self
    
    def create_manim_scene(self):
        """
        Create a Manim Scene class with the configured stream plot.
        
        Returns:
        --------
        Scene : manim.Scene
            A Manim Scene class with the configured stream plot
        """
        if self.dataset is None:
            raise ValueError("Dataset is required. Set it with set_dataset() method.")
        
        if self.background_image_path is None:
            raise ValueError("Background image path is required. Set it with set_background_image() method.")
        
        # Create a new Scene class dynamically
        class ConfiguredStreamPlot(Scene):
            def __init__(self_scene, **kwargs):
                super().__init__(**kwargs)
                self_scene.handler_ref = self  # Store reference to handler
            
            def construct(self_scene):
                # Load background image
                geo_image = ImageMobject(self.background_image_path)
                geo_image.height = self.config['image_height']
                geo_image.center()
                
                self_scene.play(FadeIn(geo_image))
                
                # Add title if specified
                if self.config['show_title']:
                    title = Text(self.config['title_text'], font_size=self.config['title_font_size'])
                    title.to_edge("UP")
                    self_scene.play(FadeIn(title))
                
                # Extract dimensions from ImageMobject
                img_width = geo_image.width
                img_height = geo_image.height
                x_min, x_max = geo_image.get_left()[0], geo_image.get_right()[0]
                y_min, y_max = geo_image.get_bottom()[1], geo_image.get_top()[1]
                
                # Extract wind data
                u10, v10, lats, lons = self._extract_wind_data()
                
                # Create interpolators
                u_interp, v_interp = self._create_interpolators(u10, v10, lats, lons)
                
                # Define wind vector function for the stream lines
                def wind_vector_func(pos):
                    x, y = pos[0], pos[1]
                    # Map from screen coordinates to geographic coordinates
                    lon = np.interp(x, [x_min, x_max], [lons.min(), lons.max()])
                    lat = np.interp(y, [y_min, y_max], [lats.min(), lats.max()])
                    
                    # Use interpolator
                    try:
                        u = u_interp([lat, lon])[0]
                        v = v_interp([lat, lon])[0]
                    except:
                        # Return zero vector if interpolation fails
                        return np.array([0.0, 0.0, 0.0])
                    
                    return np.array([u, v, 0.0])
                
                # Calculate the range of the streamline to match the ImageObject
                x_range_min = round((img_width - img_width * self.config['resize_factor']), 2) / 2
                y_range_min = round((img_height - img_height * self.config['resize_factor']), 2) / 2
                delta_y = self.config['delta_y']
                
                # Create stream lines
                stream_lines = StreamLines(
                    wind_vector_func,
                    x_range=[-1 * x_range_min, x_range_min, delta_y],
                    y_range=[-1 * y_range_min, y_range_min, delta_y],
                    color=self.config['color'],
                    dt=self.config['dt'],
                    padding=0,
                    noise_factor=self.config['noise_factor'],
                    max_anchors_per_line=self.config['max_anchors_per_line'],
                    stroke_width=self.config['stroke_width'],
                    virtual_time=self.config['virtual_time']
                )
                
                # Center the stream lines
                stream_lines.center()
                self_scene.add(stream_lines)
                
                # Animate the stream lines
                stream_lines.start_animation(
                    warm_up=False,
                    flow_speed=self.config['flow_speed'],
                    time_width=self.config['time_width']
                )
                
                self_scene.wait(self.config['animation_duration'])
        
        self.scene_class = ConfiguredStreamPlot
        return ConfiguredStreamPlot
    
    def _extract_wind_data(self):
        """
        Extract wind data (u10, v10) and coordinates (lats, lons) from the dataset.
        
        Returns:
        --------
        tuple : (u10, v10, lats, lons)
            The wind components and coordinate arrays
        """
        u10 = self.dataset['u10'].values
        v10 = self.dataset['v10'].values
        lats = self.dataset['latitude'].values
        lons = self.dataset['longitude'].values
        
        return u10, v10, lats, lons
    
    def _create_interpolators(self, u10, v10, lats, lons):
        """
        Create interpolators for the wind components.
        
        Parameters:
        -----------
        u10 : numpy.ndarray
            The u-component of wind
        v10 : numpy.ndarray
            The v-component of wind
        lats : numpy.ndarray
            The latitude coordinates
        lons : numpy.ndarray
            The longitude coordinates
        
        Returns:
        --------
        tuple : (u_interp, v_interp)
            The interpolators for u and v components
        """
        # Ensure lat is increasing and lon is increasing for interpolation
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            u10 = u10[::-1, :]
            v10 = v10[::-1, :]
        if lons[0] > lons[-1]:
            lons = lons[::-1]
            u10 = u10[:, ::-1]
            v10 = v10[:, ::-1]
        
        # Create interpolators
        u_interp = RegularGridInterpolator((lats, lons), u10)
        v_interp = RegularGridInterpolator((lats, lons), v10)
        
        return u_interp, v_interp
    
    def set_dataset(self, dataset):
        """
        Set the dataset for the stream plot.
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            The ERA5 dataset containing wind data (u10, v10)
        
        Returns:
        --------
        self : StreamPlotHandler
            Returns self for method chaining
        """
        self.dataset = dataset
        return self
    
    def set_background_image(self, image_path):
        """
        Set the background image path for the stream plot.
        
        Parameters:
        -----------
        image_path : str
            Path to the background image file
        
        Returns:
        --------
        self : StreamPlotHandler
            Returns self for method chaining
        """
        self.background_image_path = image_path
        return self
    
    def render_to_notebook(self, display_mode='video'):
        """
        Render the animation and display it directly in the notebook.
        
        Parameters:
        -----------
        display_mode : str
            How to display the animation - 'video' or 'html'
            
        Returns:
        --------
        None : Displays the animation in the notebook
        """
        if self.scene_class is None:
            self.scene_class = self.create_manim_scene()
        
        # Configure manim for notebook display
        manim_config.output_file = self.config['output_file']
        manim_config.preview = True
        
        # Render the scene
        scene = self.scene_class()
        scene.render()
        
        # Get the path to the video file
        media_dir = manim_config.get_dir("media_dir")
        quality = self.config['quality']
        video_dir = os.path.join(media_dir, "videos", self.scene_class.__name__, quality)
        video_path = os.path.join(video_dir, f"{self.config['output_file']}.mp4")
        
        if display_mode == 'video':
            return display(Video(video_path, embed=True, html_attributes="controls autoplay loop"))
        elif display_mode == 'html':
            video_tag = f"""
            <video width="{self.config['pixel_width']}" height="{self.config['pixel_height']}" controls autoplay loop>
                <source src="{video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            return display(HTML(video_tag))
        else:
            print(f"Animation rendered to: {video_path}")
    
    def preview(self, frame_number=0):
        """
        Generate a preview frame of the animation without rendering the full animation.
        
        Parameters:
        -----------
        frame_number : int
            The frame number to preview
            
        Returns:
        --------
        None : Displays a preview image in the notebook
        """
        from manim import config as manim_cfg
        from manim import tempconfig
        
        if self.scene_class is None:
            self.scene_class = self.create_manim_scene()
        
        # Use tempconfig to temporarily modify the config
        with tempconfig({"save_last_frame": True, "output_file": f"{self.config['output_file']}_preview"}):
            scene = self.scene_class()
            scene.render()
            
            # Get the path to the image file
            media_dir = manim_cfg.get_dir("media_dir")
            quality = self.config['quality']
            images_dir = os.path.join(media_dir, "images", self.scene_class.__name__)
            image_path = os.path.join(images_dir, f"{self.config['output_file']}_preview.png")
            
            # Display the image
            from IPython.display import Image
            return display(Image(image_path))
    
    def _cleanup(self):
        """Clean up temporary files if needed"""
        import shutil
        if self.temp_media_dir and os.path.exists(self.temp_media_dir):
            shutil.rmtree(self.temp_media_dir)
            
    def __del__(self):
        """Destructor to clean up resources"""
        self._cleanup()


