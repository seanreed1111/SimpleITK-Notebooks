#DicomSeriesHelper.py
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
import sys
import shutil
import subprocess
import multiprocessing
from functools import partial

# We use the multiprocess package instead of the official
# multiprocessing as it currently has several issues as discussed
# on the software carpentry page: https://hpc-carpentry.github.io/hpc-python/06-parallel/
import multiprocess as mp
import platform
import hashlib
import tempfile
import pickle
import matplotlib.pyplot as plt
import ipywidgets as widgets
import ipympl

# Maximal number of parallel processes we run.
MAX_PROCESSES = 100

def process_image(img, projection_axis, thumbnail_size):
    """
    Create a grayscale thumbnail image from the given image. If the image is 3D it is
    projected to 2D using a Maximum Intensity Projection (MIP) approach. Color images
    are converted to grayscale, and high dynamic range images are window leveled using
    a robust approach.

    Parameters
    ----------
    img (SimpleITK.Image): A 2D or 3D grayscale or sRGB image.
    projection_axis(int in [0,2]): The axis along which we project 3D images.
    thumbnail_size (list/tuple(int)): The 2D sizes of the thumbnail.

    Returns
    -------
    2D SimpleITK image with sitkUInt8 pixel type.

    """
    if (
        img.GetDimension() == 3 and img.GetSize()[2] == 1
    ):  # 2D image masquerading as 3D image
        img = img[:, :, 0]
    elif img.GetDimension() == 3:  # 3D image projected along projection_axis direction
        img = sitk.MaximumProjection(img, projection_axis)
        slc = list(img.GetSize())
        slc[projection_axis] = 0
        img = sitk.Extract(img, slc)
    if img.GetNumberOfComponentsPerPixel() == 3:  # sRGB image, convert to gray
        # Convert sRGB image to gray scale and rescale results to [0,255]
        channels = [
            sitk.VectorIndexSelectionCast(img, i, sitk.sitkFloat32)
            for i in range(img.GetNumberOfComponentsPerPixel())
        ]
        # linear mapping
        I = (
            1
            / 255.0
            * (0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])
        )
        # nonlinear gamma correction
        I = (
            I * sitk.Cast(I <= 0.0031308, sitk.sitkFloat32) * 12.92
            + I ** (1 / 2.4) * sitk.Cast(I > 0.0031308, sitk.sitkFloat32) * 1.055
            - 0.055
        )
        img = sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8)
    else:
        if img.GetPixelID() != sitk.sitkUInt8:
            # To deal with high dynamic range images that also contain outlier intensities
            # we use window-level intensity mapping and set the window:
            # to [max(Q1 - w*IQR, min_intensity), min(Q3 + w*IQR, max_intensity)]
            # IQR = Q3-Q1
            # The bounds which should exclude outliers are defined by the parameter w,
            # where 1.5 is a standard default value (same as used in box and
            # whisker plots to define whisker lengths).
            w = 1.5
            min_val, q1_val, q3_val, max_val = np.percentile(
                sitk.GetArrayViewFromImage(img).flatten(), [0, 25, 75, 100]
            )
            min_max = [
                np.max([(1.0 + w) * q1_val - w * q3_val, min_val]),
                np.min([(1.0 + w) * q3_val - w * q1_val, max_val]),
            ]
            wl_image = sitk.IntensityWindowing(
                img,
                windowMinimum=min_max[0],
                windowMaximum=min_max[1],
                outputMinimum=0.0,
                outputMaximum=255.0,
            )
            img = sitk.Cast(wl_image, sitk.sitkUInt8)
    res = sitk.Resample(
        img,
        size=thumbnail_size,
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=img.GetOrigin(),
        outputSpacing=[
            (sz - 1) * spc / (nsz - 1)
            for nsz, sz, spc in zip(thumbnail_size, img.GetSize(), img.GetSpacing())
        ],
        outputDirection=img.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=img.GetPixelID(),
    )
    res.SetOrigin([0, 0])
    res.SetSpacing([1, 1])
    res.SetDirection([1, 0, 0, 1])
    return res


def visualize_single_file(file_name, imageIO, projection_axis, thumbnail_size):
    image_file_name = ""
    image = None
    try:
        reader = sitk.ImageFileReader()
        reader.SetImageIO(imageIO)
        reader.SetFileName(file_name)
        img = reader.Execute()
        image = process_image(img, projection_axis, thumbnail_size)
        image_file_name = file_name
    except:
        pass
    return (image_file_name, image)


def visualize_files(
    root_dir, imageIO="", projection_axis=2, thumbnail_size=[128, 128], tile_size=[20, 20]
):
    """
    This function traverses the directory structure reading all user selected images
    (selction based on the image file format specified by the caller). All images are converted to 2D grayscale
    in [0,255] as follows:
    * Images with three channels are assumed to be in sRGB color space and converted to grayscale.
    * Grayscale images are window-levelled using robust values for the window-level accomodating
    * for outlying intensity values.
    * 3D images are converted to 2D using maximum intensity projection along the user specified projection axis.
    Parameters
    ----------
    root_dir (str): Path to the root of the data directory. Traverse the directory structure
                    and try to read every file as an image using the given imageIO.
    imageIO (str): Name of image IO to use. To see the list of registered image IOs use the
                   ImageFileReader::GetRegisteredImageIOs() or print an ImageFileReader.
                   The empty string indicates to read all file formats supported by SimpleITK.
    projection_axis (int in [0,2]): 3D images are converted to 2D using mean projection along the
                                    specified axis.
    thumbnail_size (2D tuple/list): The size of the 2D image tile used for visualization.
    tile_size (2D tuple/list): Number of tiles to use in x and y.

    Returns
    -------
    tuple(SimpleITK.Image, list): faux_volume comprised of tiles, file_name_list corrosponding
                                  to the image tiles.
                                  The SimpleITK image contains the meta-data 'thumbnail_size' and
                                  'tile_size'.
    """
    image_file_names = []
    faux_volume = None
    images = []

    all_file_names = []
    for dir_name, subdir_names, file_names in os.walk(root_dir):
        all_file_names += [
            os.path.join(os.path.abspath(dir_name), fname) for fname in file_names
        ]
    if platform.system() == "Windows":
        res = map(
            partial(
                visualize_single_file,
                imageIO=imageIO,
                projection_axis=projection_axis,
                thumbnail_size=thumbnail_size,
            ),
            all_file_names,
        )
    else:
        with mp.Pool(processes=MAX_PROCESSES) as pool:
            res = pool.map(
                partial(
                    visualize_single_file,
                    imageIO=imageIO,
                    projection_axis=projection_axis,
                    thumbnail_size=thumbnail_size,
                ),
                all_file_names,
            )
    res = [data for data in res if data[1] is not None]
    if res:
        image_file_names, images = zip(*res)
        if image_file_names:
            faux_volume = create_tile_volume(images, tile_size)
            faux_volume.SetMetaData(
                "thumbnail_size", " ".join([str(v) for v in thumbnail_size])
            )
            faux_volume.SetMetaData("tile_size", " ".join([str(v) for v in tile_size]))
    return (faux_volume, image_file_names)


def create_tile_volume(images, tile_size):
    """
    Create a faux-volume from a list of images. Each slice in the volume
    is constructed from tile_size[0]*tile_size[1] images. The slices are
    then joined to form the faux volume.

    Parameters
    ----------
    images (list(SimpleITK.Image(2D, sitkUInt8))): image list that we tile.
    tile_size (2D tuple/list): Number of tiles to use in x and y.

    Returns
    -------
    SimpleITK.Image(3D, sitkUInt8): Volume comprised of tiled image slices.
                                    Order of tiles matches the order of the input list.
    """
    step_size = tile_size[0] * tile_size[1]
    faux_volume = [
        sitk.Tile(images[i : i + step_size], tile_size, 0)
        for i in range(0, len(images), step_size)
    ]
    # if last tile image is smaller than others, add background content to match the size
    if len(faux_volume) > 1 and (
        faux_volume[-1].GetHeight() != faux_volume[-2].GetHeight()
        or faux_volume[-1].GetWidth() != faux_volume[-2].GetWidth()
    ):
        img = sitk.Image(faux_volume[-2]) * 0
        faux_volume[-1] = sitk.Paste(
            img, faux_volume[-1], faux_volume[-1].GetSize(), [0, 0], [0, 0]
        )
    return sitk.JoinSeries(faux_volume)


def visualize_series(
    root_dir, projection_axis=2, thumbnail_size=[128, 128], tile_size=[20, 20]
):
    """
    This function traverses the directory structure reading all DICOM series (a series can reside
    in multiple directories). All images are converted to 2D grayscale in [0,255] as follows:
    * Images with three channels are assumed to be in sRGB color space and converted to grayscale.
    * Grayscale images are window-levelled using robust values for the window-level accomodating
    * for outlying intensity values.
    * 3D images are converted to 2D using maximum intensity projection along the user specified projection axis.
    Parameters
    ----------
    root_dir (str): Path to the root of the data directory. Traverse the directory structure
                    and try to read every file as an image using the given imageIO.
    projection_axis (int in [0,2]): 3D images are converted to 2D using mean projection along the
                                    specified axis.
    thumbnail_size (2D tuple/list): The size of the 2D image tile used for visualization.
    tile_size (2D tuple/list): Number of tiles to use in x and y.

    Returns
    -------
    tuple(SimpleITK.Image, list): faux_volume comprised of tiles, series_file_name_lists corrosponding
                                  to the image tiles. The series_file_name_lists is a list of lists where
                                  the sublists are DICOM series.
                                  The SimpleITK image contains the meta-data 'thumbnail_size' and
                                  'tile_size'.
    """
    # collect the file names of all series into a dictionary with the key being
    # study:series.
    all_series_files = {}
    reader = sitk.ImageFileReader()
    for dir_name, subdir_names, file_names in os.walk(root_dir):
        sids = sitk.ImageSeriesReader_GetGDCMSeriesIDs(dir_name)
        for (
            sid
        ) in (
            sids
        ):  # Using absolute file names so that the list is valid no matter where the script is run
            file_names = [
                os.path.abspath(fname)
                for fname in sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
                    dir_name, sid
                )
            ]
            reader.SetFileName(file_names[0])
            reader.ReadImageInformation()
            study = reader.GetMetaData("0020|000d")
            key = f"{study}:{sid}"
            if key in all_series_files:
                all_series_files[key].extend(file_names)
            else:
                all_series_files[key] = list(file_names)
    images_and_files = [
        (process_series(series_data, projection_axis, thumbnail_size), series_data[1])
        for series_data in all_series_files.items()
    ]
    images, files = zip(*images_and_files)
    faux_volume = create_tile_volume(images, tile_size)
    faux_volume.SetMetaData(
        "thumbnail_size", " ".join([str(v) for v in thumbnail_size])
    )
    faux_volume.SetMetaData("tile_size", " ".join([str(v) for v in tile_size]))
    return (faux_volume, files)


def process_series(series_data, projection_axis, thumbnail_size):
    reader = sitk.ImageSeriesReader()
    _, sid = series_data[0].split(":")
    file_names = series_data[1]
    # As the files comprising a series with multiple files can reside in
    # separate directories and SimpleITK expects them to be in a single directory
    # we use a tempdir and symbolic links to enable SimpleITK to read the series as
    # a single image.
    with tempfile.TemporaryDirectory() as tmpdirname:
        if platform.system() == "Windows":
            for i, fname in enumerate(file_names):
                shutil.copy(fname, os.path.join(tmpdirname, str(i)))
        else:
            for i, fname in enumerate(file_names):
                os.symlink(fname, os.path.join(tmpdirname, str(i)))
        reader.SetFileNames(
            sitk.ImageSeriesReader_GetGDCMSeriesFileNames(tmpdirname, sid)
        )
        img = reader.Execute()
        return process_image(img, projection_axis, thumbnail_size)

# The class in the following cell `ImageSelection` provides a GUI for displaying and interacting with a tiled faux volume. The user can scroll through the faux volume "slices", zoom in, pan, and select images. When the user clicks on an image a user specified action is taken, `selection_func` is invoked with the file name(s) of the associated image. Two useful user functions are provided at the end of the code cell:
# * `show_image` - displays the original image at full resolution using an external viewer (both 2D and 3D).
# * `rm_image` - for the more confident user, delete the file(s) associated with the selected image (data cleanup).
# 
# The recommended usage is with the `show_image` ensuring that the images you selected should truly be deleted and then deleting them.

# In[37]:


class ImageSelection(object):
    def __init__(
        self, tiled_faux_vol, image_files_list, selection_func=None, figure_size=(20, 6)
    ):
        self.tiled_faux_vol = tiled_faux_vol
        self.thumbnail_size = [
            int(v) for v in self.tiled_faux_vol.GetMetaData("thumbnail_size").split()
        ]
        self.tile_size = [
            int(v) for v in self.tiled_faux_vol.GetMetaData("tile_size").split()
        ]
        self.npa = sitk.GetArrayViewFromImage(self.tiled_faux_vol)
        self.point_indexes = []
        self.selected_image_indexes = []
        self.image_files_list = image_files_list
        self.selection_func = selection_func

        ui = self.create_ui()
        display(ui)

        # Create a figure.
        self.fig, self.axes = plt.subplots(1, 1, figsize=figure_size)
        # Connect the mouse button press to the canvas (__call__ method is the invoked callback).
        self.fig.canvas.mpl_connect("button_press_event", self)

        # Display the data and the controls, first time we display the image is outside the "update_display" method
        # as that method relies on the previous zoom factor which doesn't exist yet.
        self.axes.imshow(
            self.npa[self.slice_slider.value, :, :] if self.slice_slider else self.npa,
            cmap=plt.cm.Greys_r,
        )
        self.fig.tight_layout()
        self.update_display()

    def create_ui(self):
        # Create the active GUI components. Height and width are specified in 'em' units. This is
        # a HTML size specification, size relative to current font size.
        self.viewing_checkbox = widgets.RadioButtons(
            description="Interaction mode:", options=["edit", "view"], value="edit"
        )

        self.clearlast_button = widgets.Button(
            description="Clear Last", width="7em", height="3em"
        )
        self.clearlast_button.on_click(self.clear_last)

        self.clearall_button = widgets.Button(
            description="Clear All", width="7em", height="3em"
        )
        self.clearall_button.on_click(self.clear_all)

        # Slider is only created if a 3D image, otherwise no need.
        self.slice_slider = None
        if self.npa.ndim == 3:
            self.slice_slider = widgets.IntSlider(
                description="image z slice:",
                min=0,
                max=self.npa.shape[0] - 1,
                step=1,
                value=int((self.npa.shape[0] - 1) / 2),
                width="20em",
            )
            self.slice_slider.observe(self.on_slice_slider_value_change, names="value")
            bx0 = widgets.Box(padding=7, children=[self.slice_slider])

        # Layout of GUI components. This is pure ugliness because we are not using a GUI toolkit. Layout is done
        # using the box widget and padding so that the visible GUI components are spaced nicely.
        bx1 = widgets.Box(padding=7, children=[self.viewing_checkbox])
        bx2 = widgets.Box(padding=15, children=[self.clearlast_button])
        bx3 = widgets.Box(padding=15, children=[self.clearall_button])
        return (
            widgets.HBox(children=[widgets.HBox(children=[bx1, bx2, bx3]), bx0])
            if self.slice_slider
            else widgets.HBox(children=[widgets.HBox(children=[bx1, bx2, bx3])])
        )

    def on_slice_slider_value_change(self, change):
        self.update_display()

    def update_display(self):
        # We want to keep the zoom factor which was set prior to display, so we log it before
        # clearing the axes.
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Draw the image and localized points.
        self.axes.clear()
        self.axes.imshow(
            self.npa[self.slice_slider.value, :, :] if self.slice_slider else self.npa,
            cmap=plt.cm.Greys_r,
        )
        for i, pnt in enumerate(self.point_indexes):
            if (
                self.slice_slider and int(pnt[2] + 0.5) == self.slice_slider.value
            ) or not self.slice_slider:
                self.axes.scatter(pnt[0], pnt[1], s=90, marker="+", color="yellow")
                # Get point in pixels.
        self.axes.set_title(f"selected {len(self.point_indexes)} images")
        self.axes.set_axis_off()

        # Set the zoom factor back to what it was before we cleared the axes, and rendered our data.
        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)

        self.fig.canvas.draw_idle()

    def clear_all(self, button):
        del self.point_indexes[:]
        del self.selected_image_indexes[:]
        self.update_display()

    def clear_last(self, button):
        if self.point_indexes:
            self.point_indexes.pop()
            self.selected_image_indexes.pop()
            self.update_display()

    def get_selected_images(self):
        return [self.image_files_list[index] for index in self.selected_image_indexes]

    def __call__(self, event):
        if self.viewing_checkbox.value == "edit":
            if event.inaxes == self.axes:
                x = int(round(event.xdata))
                y = int(round(event.ydata))
                z = self.slice_slider.value
                image_index = (
                    z * self.tile_size[0] * self.tile_size[1]
                    + int(y / self.thumbnail_size[1]) * self.tile_size[0]
                    + int(x / self.thumbnail_size[0])
                )
                if image_index < len(self.image_files_list):
                    # If new selection add it, otherwise just redisplay the image by calling Show.
                    if image_index not in self.selected_image_indexes:
                        self.point_indexes.append(
                            (event.xdata, event.ydata, self.slice_slider.value)
                            if self.slice_slider
                            else (event.xdata, event.ydata)
                        )
                        self.selected_image_indexes.append(image_index)
                        self.update_display()
                    if self.selection_func:
                        self.selection_func(self.image_files_list[image_index])


def show_image(image_file_name):
    if isinstance(image_file_name, str):
        img = sitk.ReadImage(image_file_name)
    else:
        # As the files comprising a DICOM series with multiple files can reside in
        # separate directories and SimpleITK expects them to be in a single directory
        # we use a tempdir and symbolic links to enable SimpleITK to read the series as
        # a single image.
        with tempfile.TemporaryDirectory() as tmpdirname:
            if platform.system() == "Windows":
                for i, fname in enumerate(image_file_name):
                    shutil.copy(
                        os.path.abspath(fname), os.path.join(tmpdirname, str(i))
                    )
            else:
                for i, fname in enumerate(image_file_name):
                    os.symlink(os.path.abspath(fname), os.path.join(tmpdirname, str(i)))
            img = sitk.ReadImage(
                sitk.ImageSeriesReader_GetGDCMSeriesFileNames(tmpdirname)
            )
    sitk.Show(img)


def rm_image(image_file_name):
    try:  # if file doesn't exist an exception is thrown.
        if isinstance(image_file_name, basestring):
            os.remove(image_file_name)
        else:
            for f in image_file_name:
                os.remove(f)
    except:
        pass
