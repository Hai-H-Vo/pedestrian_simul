# SIMULATOR


The function of each file is listed here:

**\_\_init\_\_.py**: required to make the simulator dir a package? 

**basis.py**: provides the Laguerre and Legendre basis polynomials

**dynamics.py**: general dynamics framework. *dynamics* fns returns an initializer and a step function that increments the system

**extract.py**: extract position and time data from .csv files, and compute the velocity, speed, and orientation

**render.py**: render the simulation data provided, returns an .mp4 file

**utils.py**: force and energy functions for 2-body interactions, and some data processing tools

**force.py**: force and energy functions for many-body interactions, basis representation for a general force **F**(v, d)

**environment.yml**: virtual env documentation

**setup.py**: (incomplete) makes the package downloadable

**Pedestrian_*.py**: Simulation/Extraction code for different pedestrian setups

# pedestrian_learning_main

dataloader.py and train.py modified to work with trajectory training instead of timepoint training.
