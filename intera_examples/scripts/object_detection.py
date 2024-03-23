import numpy as np
import trimesh
import os
from tqdm import tqdm
import time

from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager
from pykin.utils.kin_utils import apply_robot_to_scene
from pykin.utils.kin_utils import ShellColors as sc


class RobotCollisionDetection():
    def __init__(self) -> None:
        file_path = "urdf/sawyer/sawyer.urdf"
        self.robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0.913]), has_gripper=True, gripper_name='robotiq140')
        self.robot.setup_link_name("sawyer_base", "sawyer_right_hand")

        self.c_manager = CollisionManager(is_robot=True)
        self.c_manager.setup_robot_collision(self.robot, geom="collision")
        self.c_manager.setup_gripper_collision(self.robot, geom='collision')
        self.c_manager.show_collision_info()

        conveyor_path = '/home/yikang/catkin_ws/src/sawyer_irl_project/meshes/lab_conveyor/conveyor_assembly.stl'
        self.conveyor_mesh = trimesh.load_mesh(conveyor_path)
        self.conveyor_transform = Transform(pos=[-0.47, -0.5, -0.29], rot=[np.pi, np.pi, np.pi, np.pi])

        obstacle_path = '/home/yikang/pykin/pykin/assets/objects/meshes/bin_15.stl'
        self.obstacle_mesh = trimesh.load_mesh(obstacle_path)
        self.obstacle_transform = Transform(pos=[0.8, -0.3, 0.35], rot=[0, 0, 0])

        self.o_manager = CollisionManager()
        self.o_manager.add_object("conveyor", gtype="mesh", gparam=self.conveyor_mesh, h_mat=self.conveyor_transform.h_mat)
        self.o_manager.add_object("obstacle", gtype="mesh", gparam=self.obstacle_mesh, h_mat=self.obstacle_transform.h_mat)

        self.set_robot_pos(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        
        self.scene = trimesh.Scene()
        self.scene.add_geometry(self.conveyor_mesh, node_name="conveyor", transform=self.conveyor_transform.h_mat)
        self.scene.add_geometry(self.obstacle_mesh, node_name="obstacle", transform=self.obstacle_transform.h_mat)

    def set_robot_pos(self, pos):
        self.robot.set_transform(pos)
        for link, info in self.robot.info[self.c_manager.geom].items():
            if link in self.c_manager._objs:
                self.c_manager.set_transform(name=link, h_mat=info[3])
        # print(f'transform: {loop_start - transform_start}')
        # print(f'loop: {time.time() - loop_start}')

    def get_internal_collision(self):
        result, name = self.c_manager.in_collision_internal(return_names=True)
        return result, name
    
    def get_collision_info(self, func):
        result, name = func()
        if result:
            print(f"{sc.FAIL}Collide!! {sc.ENDC}{list(name)[0][0]} and {list(name)[0][1]}")
    
    def get_external_collision(self):
        result, name = self.c_manager.in_collision_other(self.o_manager, return_names=True)
        return result, name
    
    def get_collision_check(self, robot_pos=None):
        set_robot_start = time.time()
        if robot_pos is not None:
            self.set_robot_pos(robot_pos)
        internal_start = time.time()
        internal_result, internal_name = self.get_internal_collision()
        external_start = time.time()
        external_result, external_name = self.get_external_collision()
        # print(f'set robot: {internal_start - set_robot_start}')
        # print(f'internal: {external_start - internal_start}')
        # print(f'external: {time.time() - external_start}')
        return internal_result or external_result, internal_result, internal_name, external_result, external_name

    def render(self):
        self.scene = apply_robot_to_scene(trimesh_scene=self.scene, robot=self.robot, geom=self.c_manager.geom)
        self.scene.set_camera(np.array([np.pi / 2, 0, 0]), 3, resolution=(1024, 512))
        self.scene.show()

    def get_eef_position(self, robot_pos=None):
        if robot_pos is not None:
            self.set_robot_pos(robot_pos)

        robot_info = self.robot.info['collision']
        for link, info in robot_info.items():
            h_mat = info[3]
            if info[0] == 'sawyer_right_hand':
                axis_start = h_mat.dot(np.array([0, 0, 0, 1]))[:3]
                return axis_start
        raise ValueError('sawyer_right_hand not found')
                

if __name__ == '__main__':
    collision_detection = RobotCollisionDetection()

    # efficiency test
    # for _ in tqdm(range(1000)):
    #     robot_c_pos = np.random.uniform(low=0, high=np.pi*2, size=8)
    #     collision_detection.set_robot_pos(robot_c_pos)
    #     collision_detection.get_collision_check()

    # internal collision
    robot_c_pos = np.array([0, -0.01, -0.28, -0.39, 1.21, 0.52, 0.78, 0])
    # robot_c_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    # external collision
    # robot_c_pos = np.array([np.pi/2, 0, 0.9, 0, 0, 0, 0, 0])

    collision_detection.set_robot_pos(robot_c_pos)
    collision_detection.get_eef_position()
    collision_detection.get_collision_info(collision_detection.get_internal_collision)
    collision_detection.get_collision_info(collision_detection.get_external_collision)

    collision_detection.render()