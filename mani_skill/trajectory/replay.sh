for robot in panda 
  do
    for env_id in TeacherArmShuffle-v1 SwapThreeCubes-v1 PickPlaceThreetimes-v1 PushCubeWithSignal-v1
      do
          python mani_skill/trajectory/replay_trajectory.py \
              --traj_path="/home/chenyipeng/Memory_dependence_benchmark/data/${robot}/${env_id}/motionplanning/trajectory_${robot}.h5" \
              -o rgb \
              -c pd_ee_delta_pose \
              --save_traj \
              --num-envs 1 \
              -b physx_cpu
      done
  done

#bash mani_skill/trajectory/replay.sh

# replay
# python mani_skill/trajectory/replay_trajectory.py \
#     --traj_path="/home/chenyipeng/Memory_dependence_benchmark/data/panda/TeacherArmShuffle-v1/motionplanning/trajectory_panda.h5" \
#     -o rgb \
#     -c pd_ee_delta_pose \
#     --save_traj \
#     --num-envs 1 \
#     -b physx_cpu

