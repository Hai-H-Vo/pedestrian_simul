# IMPORTS
import numpy as onp
import jax
import jax.numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import seaborn as sns

# Tell Matplotlib how to embed animations
plt.rcParams["animation.html"] = "jshtml"  # or 'html5'

sns.set_style(style="white")

# RENDERING


def render(box_size, states, time_step, name="default", origin = (0, 0), **kwargs):
    """
    Creates a rendering of the system. Edit this to
    make it run on matplotlib.

    The Particle namedtuple has the form [R, theta, V, omega] where R is an ndarray of
    shape [particle_count, spatial_dimension], while V, theta, and omega are ndarrays of shape
    [particle_count].

    Inputs:
        box_size (float): size-length of box
        states (list(array)): list of particle positions.
        time_step (float): rendering time step, equal to dt * DELTA
        name (text): name of file to be saved
        origin (indexable): coordinates of the origin of simulation. Defaults to (0, 0)

    Extra inputs:
        extra (list(array)): list of any other particle parameter
        limits (tuple(min, max)): tuple of minimum and maximum values of above param
        periodic (bool): True if extra is a periodic param (e.g. angle). Defaults to True
        walls (list(Walls)): list of walls to render
        lines (list(Array)): list of lines to add
        size (float): size of simulated particles, measured in same unit as box_size

    Output:
        {name}.mp4 file of box state, runs at 50fps
    """
    # if states is a list (sequence of simul. frames)
    # if not isinstance(states, (onp.ndarray, jax.Array)):
    #     if not isinstance(states, list):
    #         states = [states]
    #     states = np.array(states)

    if not isinstance(states, list):
        states = [states]


    R = states

    # extra parameters to be graphed in colormap
    if 'extra' not in kwargs:
        extra = None
    else:
        extra = kwargs['extra']
        ex_min, ex_max = kwargs['limits']

        if 'periodic' not in kwargs:
            cmap = plt.cm.get_cmap('hsv')
        else:
            cmap = plt.cm.get_cmap('hsv') if kwargs['periodic'] else plt.cm.get_cmap('plasma')

        cmap.set_over('black')
        cmap.set_under('black')

    # retrieve number of frames
    # frames = R.shape[0]
    frames = len(R)

    fig, ax = plt.subplots()

    # formatting plot
    ax.set_xlim(origin[0], origin[0] + box_size)
    ax.set_ylim(origin[1], origin[1] + box_size)

    # size parameter
    if 'size' not in kwargs:
        size = 1
    else:
        primitive_size = kwargs['size']

        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width_pixels = bbox.width * fig.dpi
        height_pixels = bbox.height * fig.dpi

        x_data_per_pixel = box_size / width_pixels
        y_data_per_pixel = box_size / height_pixels

        pixel_size = primitive_size / np.mean(np.array([x_data_per_pixel, y_data_per_pixel]))
        size = (pixel_size * (72 / fig.dpi))**2 * onp.pi

    # single frame rendering
    def renderer_code(frame_num=0):
        """
        Creates an artist list of one simulation frame.
        Only works for 2D.
        """
        if frame_num == frames:
            return []

        # particles data
        curr_R = R[frame_num]
        curr_x = curr_R[:, 0]
        curr_y = curr_R[:, 1]

        # rendering: USE COLOR TO ENCODE POLARIZATION/ ANGLE OF PARTICLES.
        if extra is not None:
            curr_extra = extra[frame_num]
            particle_plot = ax.scatter(
            curr_x, curr_y, c=curr_extra, s = size, cmap=cmap, vmin=ex_min, vmax=ex_max
            )
        else:
            particle_plot = ax.scatter(
            curr_x, curr_y, s = size, c='black')

        timer = ax.text(
            0.5,
            1.05,
            f"t = {time_step * frame_num:.2f}",
            size=plt.rcParams["axes.titlesize"],
            ha="center",
            transform=ax.transAxes,
        )

        return particle_plot, timer

    artists = []
    for frame in range(frames):
        artists.append(renderer_code(frame))
        print(f'Rendered {frame + 1}/{frames} frames')

    # COLORBAR FOR EXTRA
    if extra is not None:
        fig.colorbar(artists[0][0])

    # WALL RENDERING
    if 'walls' in kwargs:
        walls = kwargs['walls']
        for wall in walls:
            start = wall.start
            end = wall.end
            x_wall = [start[0], end[0]]
            y_wall = [start[1], end[1]]
            ax.plot(x_wall, y_wall, 'k')

    # EXTRA DRAWING
    if 'lines' in kwargs:
        lines = kwargs['lines']
        for line in lines:
            ax.plot(line[:, 0], line[:, 1], '--k')

    print("Building animation...")

    # build the animation
    anim = ani.ArtistAnimation(fig, artists, interval=20, repeat_delay=1000, blit=False)

    plt.close(fig)  # keep the static PNG from appearing
    anim.save(f"{name}.mp4", writer="ffmpeg", dpi=150)

    print("Rendering complete!")
