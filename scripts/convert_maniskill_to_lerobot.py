"""
Script for converting a ManiSkill dataset to LeRobot format.
"""
import os
os.environ["HF_LEROBOT_HOME"] = "/home/chenyipeng/data/maniskill_data/lerobot_datasets/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/chenyipeng/lerobot").resolve()))

import shutil
import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
from pathlib import Path
import re
from mani_skill.utils import sapien_utils
from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json  # 用于加载 JSON 文件

REPO = "mixed_data"
ROBOT_UID_LIST = ["panda_wristcam"]
ROBOT_LIST = {"panda_wristcam": "panda"}
TASK_LIST = {"panda_wristcam": ["TeacherArmShuffle-v1","SwapThreeCubes-v1","PushCubeWithSignal-v1","PickPlaceThreetimes-v1"]}
TASK_INSTRUCTION_LIST = {
    "SwapThreeCubes-v1": "Swap the position of the bottom and middle cubes.",
    "PickPlaceThreetimes-v1": "First, pick up the red cube and place it back on the table. Next, do the same for the green cube. Finally, the blue cube.",
    "PushCubeWithSignal-v1": "Wait for the signal light to flash twice, then push the cube to the target.",
    "TeacherArmShuffle-v1": "After the cubes are swapped, pick up the cube that was originally in the middle.",
}

def load_h5_data(data):
    """
    Recursively load all HDF5 datasets into memory.
    """
    out = {}
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def main(push_to_hub: bool = False, load_count: int = 100):
    """
    Converts a ManiSkill dataset to the LeRobot format and saves it under $HF_LEROBOT_HOME.

    Args:
        dataset_file (str): The file path to the ManiSkill dataset's .h5 file.
        push_to_hub (bool): Whether to push the converted dataset to the Hugging Face Hub.
        load_count (int): The number of trajectories to load. Use -1 to load all trajectories.
    """
    for robot_name in ROBOT_UID_LIST:
        robot = ROBOT_LIST[robot_name]
        REPO_NAME = f"{REPO}/{robot_name}"  

        output_path = HF_LEROBOT_HOME / REPO_NAME
        if output_path.exists():
            shutil.rmtree(output_path)

        if robot_name in ["panda_stick_wristcam"]:
            action_dim = 6
            dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root=output_path,
                robot_type=robot_name,
                use_videos=True, 
                video_backend=None,  
                fps=10,
                features={
                    "observation.images.image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.images.wrist_image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (7,), 
                        "names": ["state"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (6,),  
                        "names": ["actions"],
                    },
                },
                image_writer_threads=24,
                image_writer_processes=12,
            )
        else:
            action_dim = 7
            dataset = LeRobotDataset.create(
                repo_id=REPO_NAME,
                root=output_path,
                robot_type=robot_name,
                use_videos=True,  
                video_backend=None,  
                fps=10,
                features={
                    "observation.images.image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.images.wrist_image": {
                        "dtype": "video",
                        "shape": (256, 256, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (8,),  
                        "names": ["state"],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (7,), 
                        "names": ["action"],
                    },
                },
                image_writer_threads=24,
                image_writer_processes=12,
            )

        task_list = TASK_LIST[robot_name]
        for task in task_list:
            dataset_file = f".data/{robot}/{task}/motionplanning/trajectory_{robot}.rgb.pd_ee_delta_pose.physx_cpu.h5"
            data = h5py.File(dataset_file, "r")
            path_obj = Path(dataset_file)
            task_name = path_obj.parent.parent.name  # set_table
            action_name = path_obj.parent.name  
            file_stem = path_obj.stem
            obj_name = re.sub(r"^\d+_", "", file_stem)
            task_desc = f"{task_name}: {action_name}_{obj_name}" 

            json_file = dataset_file.replace(".h5", ".json")
            json_data = load_json(json_file)
            episodes = json_data["episodes"]


            if load_count == -1 or load_count > len(episodes):
                load_count = len(episodes) # load_count = 200

            print(f"Loading {load_count} episodes from {dataset_file}")
            for eps_id in range(load_count):
                eps = episodes[eps_id]
                traj_key = f"traj_{eps['episode_id']}"
                if traj_key not in data:
                    continue
                trajectory = data[traj_key]
                trajectory = load_h5_data(trajectory)
                actions = np.array(trajectory["actions"], dtype=np.float32)

                obs = trajectory["obs"]

                qposs = np.array(obs["agent"]["qpos"],dtype=np.float32) # [T+1, 12]
                states = np.array(obs["extra"]["tcp_pose"],dtype=np.float32) # [T+1, 7]
                gripper = qposs[:,-1:]

                if robot_name in ["panda_stick_wristcam", "xarm6_stick_wristcam"]:
                    pass
                else:
                    states = np.concatenate((states, gripper), axis=1)

                eps_len = len(actions)
                print(f"Processing episode {eps_id} with {eps_len} frames")

                images = obs["sensor_data"]["third_view_camera"]["rgb"]
                wrist_images = obs["sensor_data"]["hand_camera"]["rgb"]
                img_dim = (len(wrist_images),len(wrist_images[0]),len(wrist_images[0][0]),len(wrist_images[0][0][0]))

                for i in range(eps_len):
                    if "third_view_camera" in obs["sensor_data"]:
                        tmp_image = images[i]
                    else:
                        tmp_image = np.zeros((256, 256, 3), dtype=np.uint8)

                    if "hand_camera" in obs["sensor_data"]:
                        tmp_wrist_image = wrist_images[i]
                    else:
                        tmp_wrist_image = np.zeros((256, 256, 3), dtype=np.uint8)
                        raise ValueError(f"Expected actions shape ({action_dim},), got {actions.shape}")

                    if "agent" in obs:
                        tmp_state = states[i]
                        # print("state:",tmp_state)
                        # tmp_state_v = qvels[i]
                    else:
                        tmp_state = np.zeros(action_dim, dtype=np.float32)
                        raise ValueError(f"Expected actions shape ({action_dim},), got {actions.shape}")
                        # tmp_state_v = np.zeros(12, dtype=np.float32)
                    tmp_action = actions[i]
                    if tmp_action.shape == (8,) and robot_name in ["widowxai_wristcam"]:
                        tmp_action = tmp_action[:7]
                    if tmp_action.shape != (action_dim,):
                        # print(tmp_action.shape)
                        raise ValueError(f"Expected actions shape ({action_dim},), got {actions.shape}")
                    # print("type of tmp_state:", type(tmp_state))
                    # print("type of actions:", type(actions))

                    # print(tmp_action)
                    dataset.add_frame(
                        frame={
                        "observation.images.image": tmp_image,
                        "observation.images.wrist_image": tmp_wrist_image,
                        "observation.state": tmp_state,
                        "action": tmp_action,
                        "task": TASK_INSTRUCTION_LIST[task_name]
                        },
                    )

                dataset.save_episode()
    if push_to_hub:
        dataset.push_to_hub(
            tags=["maniskill"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
    print("Transformation Done!")
