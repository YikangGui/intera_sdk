# import cv2
import numpy as np
import math
import random
import argparse
import os
import time

from object_detection import RobotCollisionDetection
from utils import *

# np.random.seed(0)
# random.seed(0)


JOINTS_LIMITS = {'low': np.array([-2.0, -0.75, -3.04, -0.5, -1, -0.5, -0.05]),
                 'high': np.array([1.0, 0.5, 3.04, 2, 1, 1.5, 0.05])}


class RRTNode:
    """Class to store the RRT graph"""
    def __init__(self, node, parent_node):
        self.node = node
        self.parent_node = parent_node

    # def __repr__(self) -> str:
    #     return str(f'RRT_Node({self.node.tolist()})')
    

class RRTSTARNode:
    """Class to store the RRT graph"""
    def __init__(self, node, parent_node, cost):
        self.node = node
        self.parent_node = parent_node
        self.cost = cost

    def __repr__(self) -> str:
        return str(f'RRT_Star_Node({self.node.tolist()}')
    

class RRTNodeList:
    def __init__(self, node=None):
        self.list = []
        if node:
            self.list.append(node)
    
    def append(self, node):
        self.list.append(node)

    def get_array(self):
        return np.array([i.node for i in self.list])
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, item):
         return self.list[item]


class RRT:
    def __init__(self, goal_pos, goal_range, init_pos, joints_limits=JOINTS_LIMITS, step_size=0.1, sample_goal_pos_prob=0.5):
        self.step_size = step_size
        self.init_pos = init_pos
        self.collision_detection = RobotCollisionDetection()
        self.rrt_node_list = RRTNodeList(RRTNode(self.init_pos, None))

        self.joints_limits = lambda: None
        self.joints_limits.low = joints_limits['low']
        self.joints_limits.high = joints_limits['high']

        self.goal_range = lambda: None
        self.goal_range.low = goal_range['low']
        self.goal_range.high = goal_range['high']

        self.goal_pos = goal_pos
        self.sample_goal_pos_prob = sample_goal_pos_prob

    def check_collision(self, robot_pos):
        # insert_start = time.time()
        robot_pos = np.insert(robot_pos, 0, 0)
        # print(f'insert: {time.time() - insert_start}')
        collision_check_result = self.collision_detection.get_collision_check(robot_pos)
        return collision_check_result[0]
    
    def get_nearest_node(self, sample_node):
        nearest_rrt_node = None
        _nearest_distance = 1e8

        # parallel optimization
        # paralleled_time_start = time.time()
        sample_node_tiled = np.tile(sample_node, (len(self.rrt_node_list), 1))

        distance = np.linalg.norm(sample_node_tiled - self.rrt_node_list.get_array(), axis=1)
        index = np.argmin(distance)
        nearest_rrt_node = self.rrt_node_list[index]
        nearest_distance = distance[index]
        # paralledled_time_end = time.time()

        # unparalleled_time_start = time.time()
        # for rrt_node in self.rrt_node_list:
        #     distance = np.linalg.norm(sample_node - rrt_node.node)
        #     if distance < _nearest_distance:
        #         _nearest_distance = distance
        #         _nearest_rrt_node = rrt_node  
        # unparalleled_time_end = time.time()
        # print(f'parallel: {paralledled_time_end - paralleled_time_start}\nunparallel: {unparalleled_time_end - unparalleled_time_start}')

        # assert nearest_distance == _nearest_distance 
        # assert nearest_rrt_node == _nearest_rrt_node   
           
        return nearest_rrt_node, nearest_distance
    
    def generate_new_node(self, nearest_rrt_node, sample_node, distance):
        direction = sample_node - nearest_rrt_node.node
        new_node = nearest_rrt_node.node + direction / distance * self.step_size
        return new_node
    
    def sample_points(self):
        if np.random.rand() < self.sample_goal_pos_prob:
            return self.goal_pos
        return np.random.uniform(self.joints_limits.low, self.joints_limits.high)
    
    def goal_check(self, new_node):
        if (self.goal_range.low <= new_node).all() and (new_node <= self.goal_range.high).all():
            return True

    def get_path(self):
        node = self.rrt_node_list[-1]
        path = [node]
        while node.parent_node is not None:
            parent_node = node.parent_node
            path.append(parent_node)
            node = parent_node
        return path[::-1]
    
    def optimize_path(self, path, step_size=0.1):
        new_path = [path[0]]
        optimize_finished = False

        while not optimize_finished:
            current_node = new_path[-1]
            for rest_num, node in enumerate(path[::-1]):
                # print(current_node)
                direction, distance = get_direction(current_node, node)
                check_num = math.ceil(distance / step_size)
                collision_result = False
                # print(check_num)
                for i in range(check_num):
                    check_node = current_node.node + i * step_size * direction
                    collision_result = self.check_collision(check_node)
                    if collision_result:
                        break
                if collision_result:
                    continue
                else:
                    # print(f'Already optimized {len(path) - rest_num}/{len(path)}')
                    new_path.append(node)
                    if node == path[-1]:
                        optimize_finished = True
                    break
        return new_path

    def main(self, optimized=True):
        goal_reached = False
        sample_n = 0
        max_sample_n = 1000

        if (self.joints_limits.low <= self.goal_pos).all() and (self.goal_pos <= self.joints_limits.high).all():
            pass
        else:
            print('Invalid Goal Pos')
            return False, []
        
        if (self.joints_limits.low <= self.init_pos).all() and (self.init_pos <= self.joints_limits.high).all():
            pass
        else:
            print('Invalid Init Pos')
            return False, []


        goal_collision = self.check_collision(self.goal_pos)
        print(f'Goal collision check: {goal_collision}')
        if goal_collision:
            return False, []

        while sample_n < max_sample_n and not goal_reached:
            sample_n += 1
            # 1. sample point
            sample_node = self.sample_points()

            # 2. expand current node list given sampled point
            step_2_start = time.time()
            nearest_rrt_node, nearest_distance = self.get_nearest_node(sample_node)
            new_node = self.generate_new_node(nearest_rrt_node, sample_node, nearest_distance)

            # 3. check collision & goal
            # TODO: check movement collision
            step_3a_start = time.time()
            collision_result = self.check_collision(new_node)
            step_3b_start = time.time()
            goal_result = self.goal_check(new_node)

            # 4. add to current node list
            step_4_start = time.time()
            if not collision_result:
                if goal_result:
                    new_node = RRTNode(self.goal_pos, nearest_rrt_node)
                    self.rrt_node_list.append(new_node)
                    goal_reached = True
                else:
                    # print(f'node added, current node list length: {len(self.rrt_node_list)} in {sample_n} iteration')
                    self.rrt_node_list.append(RRTNode(new_node, nearest_rrt_node))

        if goal_reached:
            print(f"Goal reached in {sample_n} iterations")
        else:
            print('Failed to reach goal')

        path = self.get_path()     
        print(f'non-optimized path len: {len(path)}')

        optimized_path = self.optimize_path(path, step_size=0.2)
        print(f'optimized path len: {len(optimized_path)}')

        if optimized:
            return goal_reached, optimized_path
        else:
            return goal_reached, path

    def get_smoothness(self, path):
        path_x, path_y, path_z, = [], [], []
        for node in path:
            robot_pos = node.node
            robot_pos = np.insert(robot_pos, 0, 0)

            x, y, z = self.collision_detection.get_eef_position(robot_pos)
            path_x.append(x)
            path_y.append(y)
            path_z.append(z)
        
        angles = []
        for i in range(len(path_x) - 2):
            a = np.array([path_x[i], path_y[i], path_z[i]])
            b = np.array([path_x[i + 1], path_y[i + 1], path_z[i + 1]])
            c = np.array([path_x[i + 2], path_y[i + 2], path_z[i + 2]])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            angles.append(angle)
        print(np.mean(angles))
        return angles
    
    def render(self, path):
        for node in path:
            self.collision_detection.set_robot_pos(np.insert(node.node, 0, 0))
            self.collision_detection.render()
    

class RRTStar(RRT):
    def __init__(self, goal_pos, goal_range, init_pos, joints_limits, step_size=0.5):
        
        self.step_size = step_size
        self.init_pos = init_pos
        self.collision_detection = RobotCollisionDetection()
        self.rrt_node_list = RRTNodeList(RRTNode(self.init_pos, None))

        self.joints_limits = lambda: None
        self.joints_limits.low = joints_limits['low']
        self.joints_limits.high = joints_limits['high']

        self.goal_range = lambda: None
        self.goal_range.low = goal_range['low']
        self.goal_range.high = goal_range['high']

        self.goal_pos = goal_pos

    def check_collision(self, robot_pos):
        # insert_start = time.time()
        robot_pos = np.insert(robot_pos, 0, 0)
        # print(f'insert: {time.time() - insert_start}')
        collision_check_result = self.collision_detection.get_collision_check(robot_pos)
        return collision_check_result[0]
    
    def get_nearest_node(self, sample_node):
        nearest_rrt_node = None
        _nearest_distance = 1e8

        # parallel optimization
        # paralleled_time_start = time.time()
        sample_node_tiled = np.tile(sample_node, (len(self.rrt_node_list), 1))

        distance = np.linalg.norm(sample_node_tiled - self.rrt_node_list.get_array(), axis=1)
        index = np.argmin(distance)
        nearest_rrt_node = self.rrt_node_list[index]
        nearest_distance = distance[index]
        # paralledled_time_end = time.time()

        # unparalleled_time_start = time.time()
        # for rrt_node in self.rrt_node_list:
        #     distance = np.linalg.norm(sample_node - rrt_node.node)
        #     if distance < _nearest_distance:
        #         _nearest_distance = distance
        #         _nearest_rrt_node = rrt_node  
        # unparalleled_time_end = time.time()
        # print(f'parallel: {paralledled_time_end - paralleled_time_start}\nunparallel: {unparalleled_time_end - unparalleled_time_start}')

        # assert nearest_distance == _nearest_distance 
        # assert nearest_rrt_node == _nearest_rrt_node   
           
        return nearest_rrt_node, nearest_distance
    
    def generate_new_node(self, nearest_rrt_node, sample_node, distance):
        direction = sample_node - nearest_rrt_node.node
        new_node = nearest_rrt_node.node + direction / distance * self.step_size
        return new_node
    
    def sample_points(self):
        if np.random.rand() < 0.2:
            return self.goal_pos
        return np.random.uniform(self.joints_limits.low, self.joints_limits.high)
    
    def goal_check(self, new_node):
        if (self.goal_range.low <= new_node).all() and (new_node <= self.goal_range.high).all():
            return True

    def get_path(self):
        node = self.rrt_node_list[-1]
        path = [node]
        while node.parent_node is not None:
            parent_node = node.parent_node
            path.append(parent_node)
            node = parent_node

    def get_smothness(self, path):
        for node in path:
            self.collision_detection.get_eef_position()
        return path[::-1]

    def main(self):
        goal_reached = False
        sample_n = 0
        max_sample_n = 1000

        while sample_n < max_sample_n and not goal_reached:
            sample_n += 1
            # 1. sample point
            sample_node = self.sample_points()

            # 2. expand current node list given sampled point
            step_2_start = time.time()
            nearest_rrt_node, nearest_distance = self.get_nearest_node(sample_node)
            new_node = self.generate_new_node(nearest_rrt_node, sample_node, nearest_distance)

            # 3. check collision & goal
            # TODO: check movement collision
            step_3a_start = time.time()
            collision_result = self.check_collision(new_node)
            step_3b_start = time.time()
            goal_result = self.goal_check(new_node)

            # 4. add to current node list
            step_4_start = time.time()
            if not collision_result:
                if goal_result:
                    new_node = RRTNode(self.goal_pos, nearest_rrt_node)
                    self.rrt_node_list.append(new_node)
                    goal_reached = True
                    print(f"Goal reached in {sample_n} iterations")
                else:
                    # print(f'node added, current node list length: {len(self.rrt_node_list)} in {sample_n} iteration')
                    self.rrt_node_list.append(RRTNode(new_node, nearest_rrt_node))

            # print(f'step 2: {step_3a_start - step_2_start}')
            # print(f'step 3a: {step_3b_start - step_3a_start}')
            # print(f'step 3b: {step_4_start - step_3b_start}')
        
        path = self.get_path()
        return goal_reached, path
    
    def render(self, path):
        for node in path:
            self.collision_detection.set_robot_pos(np.insert(node.node, 0, 0))
            self.collision_detection.render()
 

class RRTConnect(RRT):
    def __init__(self, goal_pos, goal_range, init_pos, joints_limits, step_size=0.1):
        super().__init__(goal_pos, goal_range, init_pos, joints_limits, step_size)
        self.rrt_node_list_goal = RRTNodeList(RRTNode(self.goal_pos, None))
        self.rrt_node_list_init = RRTNodeList(RRTNode(self.init_pos, None))
        del self.rrt_node_list
        self.sample_goal_pos_prob = 0
        self.closest_distance = 1e10

    def get_nearest_node(self, sample_node):
        sample_goal_node_tiled = np.tile(sample_node, (len(self.rrt_node_list_goal), 1))
        distance_goal = np.linalg.norm(sample_goal_node_tiled - self.rrt_node_list_goal.get_array(), axis=1)
        index_goal = np.argmin(distance_goal)
        nearest_rrt_node_goal = self.rrt_node_list_goal[index_goal]
        nearest_distance_goal = distance_goal[index_goal]

        sample_init_node_tiled = np.tile(sample_node, (len(self.rrt_node_list_init), 1))
        distance_init = np.linalg.norm(sample_init_node_tiled - self.rrt_node_list_init.get_array(), axis=1)
        index_init = np.argmin(distance_init)
        nearest_rrt_node_init = self.rrt_node_list_init[index_init]
        nearest_distance_init = distance_init[index_init]

        return (nearest_rrt_node_goal, nearest_distance_goal), (nearest_rrt_node_init, nearest_distance_init)
    
    def connectivity_check_node_node(self, node1, node2, check_threshold=0.5):
        distance = np.sqrt(((node1 - node2) ** 2).sum())
        self.closest_distance = min(distance, self.closest_distance)
        print(f'node node: {distance}, closest:{self.closest_distance}')
        return distance < check_threshold
    
    def connectivity_check_node_tree(self, node, tree, check_threshold=0.5):
        node_tiled = np.tile(node, (len(tree), 1))
        distance = np.linalg.norm(node_tiled - tree.get_array(), axis=1)
        index = np.argmin(distance)
        nearest_node = tree[index]
        nearest_distance = distance[index]
        self.closest_distance = min(nearest_distance, self.closest_distance)
        print(f'node tree: {nearest_distance}, closest:{self.closest_distance}')
        return nearest_distance < check_threshold, nearest_node, nearest_distance
    
    def main(self):
        goal_reached = False
        sample_n = 0
        max_sample_n = 1000

        while sample_n < max_sample_n and not goal_reached:
            print(sample_n)
            sample_n += 1
            # 1. sample point
            sample_node = self.sample_points()

            # 2. expand current node list given sampled point
            (nearest_rrt_node_goal, nearest_distance_goal), (nearest_rrt_node_init, nearest_distance_init) = self.get_nearest_node(sample_node)
            new_node_goal = self.generate_new_node(nearest_rrt_node_goal, sample_node, nearest_distance_goal)
            new_node_init = self.generate_new_node(nearest_rrt_node_init, sample_node, nearest_distance_init)

            # 3. check collision & goal
            # TODO: check movement collision
            collision_result_goal = self.check_collision(new_node_goal)
            collision_result_init = self.check_collision(new_node_init)

            if not collision_result_goal:
                new_goal_rrt_node = RRTNode(new_node_goal, nearest_rrt_node_goal)
                self.rrt_node_list_goal.append(new_goal_rrt_node)

            if not collision_result_init:
                new_init_rrt_node = RRTNode(new_node_init, nearest_rrt_node_init)
                self.rrt_node_list_init.append(new_init_rrt_node)

            if not collision_result_goal and not collision_result_init:
                if self.connectivity_check_node_node(new_node_goal, new_node_init):
                    goal_reached = True
                    self.rrt_node_list_init.append(RRTNode(new_node_goal, new_init_rrt_node))
                    self.rrt_node_list_goal.append(RRTNode(new_node_init, new_goal_rrt_node))
                    print(f"Goal reached in {sample_n} iterations")
                
            if collision_result_goal and not collision_result_init:
                check_result, node, _ = self.connectivity_check_node_tree(new_node_init, self.rrt_node_list_goal)
                if check_result:
                    goal_reached = True
                    self.rrt_node_list_goal.append(RRTNode(new_node_init, node))
                    self.rrt_node_list_init.append(RRTNode(node.node, new_init_rrt_node))
                    print(f"Goal reached in {sample_n} iterations")

            if not collision_result_goal and collision_result_init:
                check_result, node, _ = self.connectivity_check_node_tree(new_node_goal, self.rrt_node_list_init)
                if check_result:
                    goal_reached = True
                    self.rrt_node_list_init.append(RRTNode(new_node_goal, node))
                    self.rrt_node_list_goal.append(RRTNode(node.node, new_goal_rrt_node))
                    print(f"Goal reached in {sample_n} iterations")

        return goal_reached, self.get_path()

if __name__ == '__main__':
    goal_pos = np.array([-2.10, 1.21, -2.34, 1.63, -1.11, -2.96, 0])
    init_pos = np.array([0.28, 0.33, -1.30, 1.31, 1.83, 1.24, 0])
    goal_range = {'low': goal_pos - 0.1,
                  'high': goal_pos + 0.1}
    joints_limits = {'low': np.array([-3.05, -3.8, -3.04, -3.04, -2.97, -2.97, -4.71]),
                     'high': np.array([3.05, 2.27, 3.04, 3.04, 2.97, 2.97, 4.71])}

    path_planner = RRT(goal_pos=goal_pos, init_pos=init_pos, goal_range=goal_range, joints_limits=joints_limits, step_size=0.1)
    # path_planner = RRTConnect(goal_pos=goal_pos, init_pos=init_pos, goal_range=goal_range, joints_limits=joints_limits, step_size=0.2)
    goal_reached, path = path_planner.main()
    # path_planner.get_smoothness(path)
    # print(len(path))
    # path_planner.render(path)