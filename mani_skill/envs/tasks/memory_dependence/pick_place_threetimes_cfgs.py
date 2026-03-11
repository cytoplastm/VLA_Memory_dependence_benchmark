"""
SwapThreeCubes-v1 configs.
This file defines the parameters for the SwapThreeCubes task for different robots.
"""

PICK_PLACE_THREETIMES_CONFIGS = {
    "panda": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,

        "sensor_cam_eye_pos": [0.3,0,0.6,],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.1, 0, 0.1],

        "human_cam_eye_pos": [0.6,0.7,0.6,],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "fetch": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
}
