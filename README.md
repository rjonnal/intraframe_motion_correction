# Intra-frame motion correction using lag biases

These scripts demonstrate an approach for correcting intra-frame motion in raster-scanned images (such as adaptive optics (AO) scanning light ophthalmoscope (SLO) or optical coherence tomography (OCT) images).

## Preliminaries

Verify installation of:

1. [python 2.7](https://www.python.org/download/releases/2.7/)

2. [numpy](http://numpy.org)

3. [matplotlib](http://matplotlib.org)

4. [scipy](https://www.scipy.org/)

A simple way to do this to install the python 2.7 version of the [Anaconda python distribution](https://www.anaconda.com/distribution/), which installs the correct versions of all the prerequisites above.

## Contents of this repository

### SLO image files

1. `slo_frames_real_large/`: A series of 100 images acquired with the [UC Davis AO-SLO](http://vsri.ucdavis.edu). Each image subtends 2 degrees horizontally and vertically, with the upper left corner of the image at the foveal center.

1. `slo_frames_real_small/`: Cropped versions of the images in `slo_frames_real_large/`, containing the lower right quadrant (1 deg x 1 deg) of each.

1. `slo_frames_simulated/`: A series of simulated motion-affected AO-SLO images, created using `create_simulated_images.py`, described below, and data in the `object/` and `simulated_eye_traces/` directories.

1. `slo_frames_simulated_idiosyncratic/`: A series of simulated motion-affected AO-SLO images, created using `create_simulated_images.py`, described below, and data in the `object/` and `simulated_eye_traces/` directories. In this series, idiosyncratic eye movements are added to the simulated eye movement traces.

### Python scripts

1. `demonstrate_registration.py`: This script performs strip-based registration of the real or simulated AO-SLO frames described above, as well as the intra-frame motion estimation and correction step described in Azimipour, 2018. This script demonstrates all of the key concepts described in Azimipour, 2018, and contains references to equations presented in the paper.

1. `create_simulated_images.py`: The script used to generate the simulated AO-SLO images described above.

### Other data

1. `object/`: A simulated retinal mosaic, generated as described in Azimipour, 2018.

1. `simulated_eye_traces/`: Simulated eye traces for 200 AO-SLO frames acquired at 30 Hz,as described in Azimipour, 2018, following the self-avoiding walk model described in Engbert, 2011.

1. `tmp`: A temporary directory that stores intermediate data created by `demonstrate_registration.py`. To start the registration process from the beginning, delete this directory or files within it accordingly.

1. `README.md`: This document.

## References

Azimipour, 2018: Azimipour et al., PLoS One, 2018

Engbert, 2011: Engbert et al., PNAS, 2011
