import csv
import numpy as onp
import jax.numpy as np
import pandas as pd
from jax import vmap
from functools import partial
from .utils import special_sme

def extract(filename, save=False, savefilename=None, smooth=False, timeframe=0) -> dict:
    """
    Extract the position data of pedestrians from a .csv file.
    Data will be used for running simul video. All unrecorded positions and orientations will be
    denoted as 999. All unrecorded velocities and speeds will be denoted as 0.

    Arguments:
        filename (str): Name of the .npz savefile
        save (bool): True to save the extracted data to an .npz file of same name. Defaults to False
        savefilename (str): Save file location. Defaults to filename
        smooth (bool): True to return a moving average of the position data. Defaults to False
        timeframe (int): Number of frames included in the moving average. Defaults to 0

    Returns:
        A dict with 5 keys (positions, velocities, speeds, orientations, time_step).
        positions = array of arrays of position arrays
        velocities = array of arrays of velocity arrays
        speeds = array of arrays of speeds
        orientations = array of arrays of orientations
        time_step = float of time between datapoints
    """
    if savefilename is None:
        savefilename = filename

    if smooth is True:
        savefilename = f"{savefilename}_smooth"

    NULL = np.array([0, 0])
    INFTY = np.array([999, 999])

    print("Begin extraction...")

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

    parsed_pos = np.array(positions)

    # smoothing out the positions
    if smooth:
        smoothed_pos = np.apply_along_axis(special_sme, 0, parsed_pos, sum_num=timeframe, null=999)
        front = int((timeframe + 1) / 2)
        back = int(-(timeframe - 1) / 2)
        parsed_pos = np.concatenate((parsed_pos[:front],
                                    smoothed_pos[front : back],
                                    parsed_pos[back:]))

    print("Position data extracted...")

    # velocity recovery
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

        far = np.select((np.logical_and(first, final), first, final),
                        (NULL, (pos4 - pos2) / 2, (pos2 - pos0) / 2),
                        (pos4 - pos0) / 4)
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

    print("Velocity data extracted...")

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

    print("Speed and orientation data extracted...")

    # save file
    if save:
        print("Saving file...")
        np.savez(savefilename,
                 positions=parsed_pos,
                 velocities=velocities,
                 speeds=speeds,
                 orientations=orientations,
                 time_step=time_step)

    print("Extraction complete!")

    return {'time_step' : time_step,
            'positions' : parsed_pos,
            'velocities' : velocities,
            'speeds' : speeds,
            'orientations' : orientations}

def load_hermes(path, savepath=None, **kwargs):
    """
    Write the data from HERMES dataset into the compatible .csv file. HERMES files have a
    timestep of 1/16 second

    Arguments:
        path (str): Path to the HERMES file

    Returns:
        None
    """
    if savepath is None:
        name = path.rstrip(".txt")
        savepath = f"{name}.csv"

    csv_columns = ["agent_id", "frame_id", "pos_x", "pos_y", "pos_z"]
    # read from csv => fill traj table
    raw_dataset = pd.read_csv(path, sep=r"\s+", header=None, names=csv_columns)

    print("HERMES dataset acquired...")

    # convert from cm => meter
    raw_dataset["pos_x"] = raw_dataset["pos_x"] / 100.
    raw_dataset["pos_y"] = raw_dataset["pos_y"] / 100.

    # ALT METHOD - nice try
    # all_ids = list(set(raw_dataset["agent_id"]))
    # data_for_each_id = {}
    # for id in all_ids:
    #     data_for_each_id[id] = raw_dataset[raw_dataset["agent_id"] == id][
    #         ["frame_id", "pos_x", "pos_y"]
    #     ].copy()

    # df_data = pd.DataFrame()
    # all_frames = sorted(list(set(raw_dataset["frame_id"])))

    # df_data["frame_id"] = all_frames

    # for id, data in data_for_each_id.items():
    #     data = data.rename(
    #         columns = {
    #             "pos_x": f"X_{id}",
    #             "pos_y": f"Y_{id}"
    #         }
    #     )
    #     df_data = df_data.merge(data, how="left", on="frame_id")

    # df_data = df_data.replace(np.nan, 0)

    # df_data["frame_id"] = (df_data["frame_id"] - df_data["frame_id"].min()) / 16
    # df_data = df_data.rename(
    #     columns={"frame_id": "T"}
    # )

    # df_data = df_data.set_index('T')
    # df_data.to_csv(savepath)

    # retrieve number of people & frames:
    N = raw_dataset['agent_id'].max()
    start_frame = raw_dataset['frame_id'].min()
    end_frame = raw_dataset['frame_id'].max()

    # time data:
    time_data = onp.array([1 / 16 * n for n in range(end_frame - start_frame + 1)])

    # positional data:
    def pos_data(n):
        agent_id = n // 2 + 1

        raw_agent_dataset = raw_dataset[raw_dataset["agent_id"] == agent_id]
        agent_dataset = [0 for _ in range(end_frame - start_frame + 1)]

        if n % 2:
            for frame, elt in zip(raw_agent_dataset['frame_id'], raw_agent_dataset['pos_y']):
                agent_dataset[frame - start_frame] = elt
        else:
            for frame, elt in zip(raw_agent_dataset['frame_id'], raw_agent_dataset['pos_x']):
                agent_dataset[frame - start_frame] = elt
        return agent_dataset

    # convert
    traj_dataset = pd.DataFrame({"T" : time_data} |
                                {(f"X_{n // 2 + 1}" if n % 2 == 0 else f"Y_{n // 2 + 1}") : pos_data(n)
                                 for n in range(2 * N)})

    traj_dataset.set_index("T", inplace=True)
    traj_dataset.to_csv(savepath)

    print("Conversion complete!")

def load_shibuya(path, savepath=None, **kwargs):
    """
    Write the data from the Shibuya dataset into the compatible .csv file. Shibuya files have a
    timestep of 1/30 second

    Arguments:
        path (str): Path to the Shibuya file

    Returns:
        None
    """
    if savepath is None:
        name = path.rstrip(".txt")
        savepath = f"{name}.csv"

    # read from csv => fill traj table
    raw_dataset = pd.read_csv(path, sep=r'\s+')
    print("Shibuya dataset acquired...")

    # retrieve number of people & frames:
    N = raw_dataset['new_ID'].max()
    start_frame = raw_dataset['t'].min()
    end_frame = raw_dataset['t'].max()

    # time data:
    time_data = onp.array([1 / 30 * n for n in range(end_frame - start_frame + 1)])

    # positional data:
    def pos_data(n):
        agent_id = n // 2 + 1

        raw_agent_dataset = raw_dataset[raw_dataset["new_ID"] == agent_id]
        agent_dataset = [0 for _ in range(end_frame - start_frame + 1)]

        if n % 2:
            for frame, elt in zip(raw_agent_dataset['t'], raw_agent_dataset['y']):
                agent_dataset[frame - start_frame] = elt
        else:
            for frame, elt in zip(raw_agent_dataset['t'], raw_agent_dataset['x']):
                agent_dataset[frame - start_frame] = elt
        return agent_dataset

    # convert
    traj_dataset = pd.DataFrame({"T" : time_data} |
                                {(f"X_{n // 2 + 1}" if n % 2 == 0 else f"Y_{n // 2 + 1}") : pos_data(n)
                                 for n in range(2 * N)})

    traj_dataset.set_index("T", inplace=True)
    traj_dataset.to_csv(savepath)

    print("Conversion complete!")
