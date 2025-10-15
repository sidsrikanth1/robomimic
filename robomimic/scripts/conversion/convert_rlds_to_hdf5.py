"""

SimplerEnv rain:
python train.py --config /opt/robomimic/robomimic/exps/simpler_env.json

CokeCan:

"""

import os
import h5py
import json
import argparse
import numpy as np

# import gym
import robomimic
import robomimic.envs.env_base as EB
from robomimic.utils.log_utils import custom_tqdm

import tensorflow_datasets as tfds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="rlds env name",
        default="kinova_coke_push",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/opt/robomimic/datasets/cokecan_rlds_data.hdf5",
        # default="/home/ssrikant/hdf5_datasets/cokecan_rlds_data.hdf5",
        help="output dir"
    )
    args = parser.parse_args()

    # base_folder = args.folder
    # if base_folder is None:
    #     base_folder = os.path.join(robomimic.__path__[0], "../datasets")
    # base_folder = os.path.join(base_folder, "d4rl")
    base_folder = os.path.dirname(args.output)
    
    # raw_dataset_name = "kinova_coke_push"
    # raw_dataset_name = "simpler_env_success_dataset"
    raw_dataset_name = args.dataset
    
    data_dir = "/tensorflow_datasets/"
    raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    
    # open hdf5 [TESTING]
            # ds = h5py.File("/opt/robomimic/datasets/lift/ph/low_dim_v15.hdf5", "r")
            # ds = h5py.File("/home/ssrikant/hdf5_datasets/converted/cokecan.hdf5", "r")
            
            # # print(ds)
            # # print(ds.keys())
            # print(ds['data']['demo_0']['actions'])
            
            
            # print(
            #     ds['data']['demo_0']['actions'].shape,
            # )
            

    # check dones (ds['data']['demo_0']['dones'])
    # dones = ds['data']['demo_0']['dones']
    
    # hdf5 output file
    write_folder = os.path.join(base_folder, "converted")
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    output_path = os.path.join(base_folder, "converted", f"{raw_dataset_name}.hdf5")
    f_sars = h5py.File(output_path, "w")
    f_sars_grp = f_sars.create_group("data")

    
    ctr = 0
    total_samples = 0
    num_traj = 0
    traj = dict(obs=[], next_obs=[], actions=[], rewards=[], dones=[], state=[])


    print("\nConverting hdf5...") # kinova_coke_push
    for episode in raw_dataset:
        # print(episode)
        
        obs = []
        wrist_obs = []
        state = []
        actions = []
        task = []
        rewards = []
        dones = []
        
        # iterate through and create the list appropriately
        for step in episode["steps"].as_numpy_iterator():
            obs.append(step["observation"]["image"])
            state.append(step["observation"]["proprio"])
            actions.append(step["action"])
            task.append(step["language_instruction"].decode())
            
            if "wrist_image" in step["observation"]:
                wrist_obs.append(step["observation"]["wrist_image"])
            
            dones.append(0)
            ctr += 1
        
        dones[-1] = 1  # mark last step as done
        rewards = [0] * len(dones)  # dummy rewards
        rewards[-1] = 1  # mark last step reward as 1
        
        
        # next obs is just offset by 1, last next_obs is copy of last obs (same for state)
        next_obs = obs[1:] + [obs[-1]] 
        next_state = state[1:] + [state[-1]]
        
        traj["obs"] = obs
        traj["next_obs"] = next_obs
        traj["actions"] = actions
        traj["rewards"] = rewards
        traj["dones"] = dones
        traj["state"] = state
        traj["next_state"] = next_state
        
        # "robot0_eef_pos",
        # "robot0_eef_quat",
        # "robot0_gripper_qpos"
        
        # store trajectory
        print("Storing trajectory of length {}...".format(len(traj["actions"])))
        ep_data_grp = f_sars_grp.create_group("demo_{}".format(num_traj))
        
        # obs
        ep_data_grp.create_dataset("obs/agentview_image", data=np.array(traj["obs"]))
        ep_data_grp.create_dataset("obs/robot0_eef_pos", data=np.array(traj["state"])[:, :3])
        ep_data_grp.create_dataset("obs/robot0_eef_quat", data=np.array(traj["state"])[:, 3:7])
        ep_data_grp.create_dataset("obs/robot0_gripper_qpos", data=np.array(traj["state"])[:, 7:])
        
        # next obs
        ep_data_grp.create_dataset("next_obs/agentview_image", data=np.array(traj["next_obs"]))
        ep_data_grp.create_dataset("next_obs/robot0_eef_pos", data=np.array(traj["next_state"])[:, :3])
        ep_data_grp.create_dataset("next_obs/robot0_eef_quat", data=np.array(traj["next_state"])[:, 3:7])
        ep_data_grp.create_dataset("next_obs/robot0_gripper_qpos", data=np.array(traj["next_state"])[:, 7:])
        
        # check the wrist image
        if len(wrist_obs) > 0:
            next_wrist_obs = wrist_obs[1:] + [wrist_obs[-1]]

            ep_data_grp.create_dataset("obs/wrist_obs", data=np.array(wrist_obs))
            ep_data_grp.create_dataset("next_obs/wrist_obs", data=np.array(next_wrist_obs))
            
        
        
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        ep_data_grp.attrs["num_samples"] = len(traj["actions"])
        total_samples += len(traj["actions"])
        num_traj += 1

        # reset
        ctr = 0
        traj = dict(obs=[], next_obs=[], actions=[], rewards=[], dones=[], state=[])


    print("\nExcluding {} samples at end of file due to no trajectory truncation.".format(len(traj["actions"])))
    print("Wrote {} trajectories to new converted hdf5 at {}\n".format(num_traj, output_path))

    # create env meta data
    env_meta = dict()
    env_meta["type"] = EB.EnvType.ROBOSUITE_TYPE
    env_meta["env_name"] = ("SimplerEnvTeleop")
    # hardcode robosuite v0.3 args
    robosuite_args = {
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": False,
        "camera_depth": False,
        "camera_height": 256,
        "camera_width": 256,
        "camera_name": "",
        "gripper_visualization": False,
        "reward_shaping": False,
        "control_freq": 100,
    }
    env_meta["env_kwargs"] = robosuite_args
    
    
    # metadata
    f_sars_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)
    f_sars_grp.attrs["total"] = total_samples
    
    
    # filter keys
    mask_grp = f_sars.create_group("mask")
    demo_names = list(f_sars_grp.keys())
    mask_grp.create_dataset("train", data=np.array(demo_names, dtype='S'))
    mask_grp.create_dataset("valid", data=np.array([], dtype='S'))

    f_sars.close()