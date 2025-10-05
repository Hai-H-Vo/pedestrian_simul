import csv
import numpy as onp
import jax.numpy as np
from jax import vmap

def extract(filename, save=False):
    """
    Extract the position data of pedestrians from a .csv file.
    Data will be used for running simul video. All unrecorded positions and orientations will be
    denoted as 999. All unrecorded velocities and speeds will be denoted as 0.

    Inputs:
        filename (str)
        save (bool): True to save the extracted data to an .npz file of same name. Defaults to False

    Output:
        dictionary of 5 params:
        positions = array of arrays of position arrays
        velocities = array of arrays of velocity arrays
        speeds = array of arrays of speeds
        orientations = array of arrays of orientations
        time_step = float of time between datapoints
    """
    NULL = np.array([0, 0])
    INFTY = np.array([999, 999])

    # collecting positions and time step
    with open(f'{filename}.csv', 'r') as f:
        csv_reader = csv.reader(f)

        times = []
        positions = []

        next(csv_reader)
        for row in csv_reader:
            temp_positions = []

            coords = []
            for i, entry in enumerate(row):
                parsed_entry = float(entry)

                if i == 0:
                    times.append(parsed_entry)
                else:
                    coords.append(parsed_entry)

                # print(coords)
                # consider saving nonexistent results as False ==> use bool np.where method to search
                if i % 2 == 0 and i != 0:
                    if not (coords[0] or coords[1]):
                        temp_positions.append(INFTY)
                    else:
                        temp_positions.append(np.array(coords))

                    coords = []

            positions.append(np.array(temp_positions))
            temp_positions = []

        time_step = times[-1] / (len(times) - 1)

    # velocity recovery
    parsed_pos = np.array(positions)

    def _dpos_calc(pos0, pos1, pos2, pos3, pos4):
        # pos0: pastpast pos
        # pos1: past pos
        # pos2: curr pos
        # pos3: next pos
        # pos4: nextnext pos
        # curr input: (2, )
        first = np.array_equal(pos0, INFTY)
        prev = np.array_equal(pos1, INFTY)
        curr = np.array_equal(pos2, INFTY)
        post = np.array_equal(pos3, INFTY)
        final = np.array_equal(pos4, INFTY)

        far = np.select((np.logical_and(first, final), first, final), (NULL, (pos4 - pos2) / 2, (pos2 - pos0) / 2), (pos4 - pos0) / 4)
        near = np.select((prev, post), (pos3 - pos2, pos2 - pos1), (pos3 - pos1) / 2)

        return np.select(condlist=(curr, np.logical_and(prev, post)),
                        choicelist=(NULL, far),
                        default=near)

    # vmap _dpos_calc to work with (N_time, N_particles, 2, ) matrices
    def dpos_calc(positions):
        # parse positions
        particle_num = np.shape(positions)[1]
        FULL_INFTY = np.array([[[999, 999]] * particle_num])
        first_positions = np.concatenate((FULL_INFTY, FULL_INFTY, positions[:-2]))
        prev_positions = np.concatenate((FULL_INFTY, positions[:-1]))
        post_positions = np.concatenate((positions[1:], FULL_INFTY))
        final_positions = np.concatenate((positions[2:], FULL_INFTY, FULL_INFTY))

        return vmap(vmap(_dpos_calc))(first_positions, prev_positions, positions, post_positions, final_positions)

    velocities = dpos_calc(parsed_pos) / time_step

    # speeds and orientations recovery
    # endgoal: 2 arrays of size (N_time, N_particles, )
    def _speed_ori(velocity):
        return np.where(np.array_equal(velocity, NULL), np.array([0, 999]),
                        np.array([np.linalg.norm(velocity), np.atan2(velocity[1], velocity[0])]))

    def speed_ori(velocities):
        return vmap(vmap(_speed_ori))(velocities)

    speeds_oris = speed_ori(velocities)
    speeds = speeds_oris[:, :, 0]
    orientations = speeds_oris[:, :, 1]

    # save file
    if save:
        np.savez(filename,
                 positions=parsed_pos,
                 velocities=velocities,
                 speeds=speeds,
                 orientations=orientations,
                 time_step=time_step)

    return {'time_step' : time_step,
            'positions' : positions,
            'velocities' : velocities,
            'speeds' : speeds,
            'orientations' : orientations}
