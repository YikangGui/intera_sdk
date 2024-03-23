#! /usr/bin/env python
# Copyright (c) 2016-2018, Rethink Robotics Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rospy
import numpy as np
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_interface import Limb

from rrt import RRT

def main():
    try:
        rospy.init_node('go_to_joint_angles_py')
        limb = Limb()
        traj = MotionTrajectory(limb = limb)

        wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=0.5,
                                         max_joint_accel=0.5,
                                         joint_tolerances=0.05)
        waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)

        joint_angles = limb.joint_ordered_angles()

        waypoint.set_joint_angles(joint_angles = joint_angles)
        traj.append_waypoint(waypoint.to_msg())

        zero_pos = np.array([0, 0, 0, 0, 0, 0, 0])

        # if result is None:
        #     rospy.logerr('Trajectory FAILED to send')
        #     return

        # if result.result:
        #     rospy.loginfo('Motion controller successfully finished the trajectory!')
        # else:
        #     rospy.logerr('Motion controller failed to complete the trajectory with error %s',
        #                  result.errorId)

        # home     
        home_pos = np.array([0.00, -0.53, -0.46, 1.29, 0.51, 0.94, 0])

        # bin     
        bin_pos = np.array([-1.52, -0.49, -0.43, 1.50, 0.50, 0.71, 0])

        # pick     
        pick_pos = np.array([-0.01, -0.28, -0.39, 1.21, 0.52, 0.78, 0])
        # pick_pos = np.array([-0.31, 0.01, -0.83, 0.89, 1.01, 0.97, 0])

        # random hard pos     
        hard_pos1 = np.array([1, 0.4, -0.61, 0.66, 0.19, 0.11, 0])
        
        # current pos
        current_pos = np.array(joint_angles)

        # test pos
        test_pos = np.array([0, -0.45, -0.27, 0.8, 0.18, 1, 0])

        init_pos = home_pos
        goal_pos = home_pos
        
        print(f'Init pos: {init_pos}')
        print(f'Goal pos: {goal_pos}')

        waypoint.set_joint_angles(joint_angles = init_pos)
        traj.append_waypoint(waypoint.to_msg())

        result = traj.send_trajectory(timeout=None)

        goal_range = {'low': goal_pos - 0.15,
                      'high': goal_pos + 0.15}
        path_planner = RRT(goal_pos=goal_pos, init_pos=init_pos, goal_range=goal_range, step_size=0.1)
        goal_reached, path = path_planner.main(optimized=True)

        print(f'Goal Reached: {goal_reached}')
        if goal_reached:
            traj = MotionTrajectory(limb = limb)
            wpt_opts = MotionWaypointOptions(max_joint_speed_ratio=0.5,
                                            max_joint_accel=0.5)
            waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)
            for node in path:
                waypoint.set_joint_angles(joint_angles = node.node.tolist())
                traj.append_waypoint(waypoint.to_msg())
            
            result = traj.send_trajectory(timeout=None)
            if result is None:
                rospy.logerr('Trajectory FAILED to send')
                return

            if result.result:
                rospy.loginfo('Motion controller successfully finished the trajectory!')
            else:
                rospy.logerr('Motion controller failed to complete the trajectory with error %s',
                            result.errorId)

    except rospy.ROSInterruptException:
        rospy.logerr('Keyboard interrupt detected from the user. Exiting before trajectory completion.')


if __name__ == '__main__':
    main()

    # optimize the solution and reduce the jerkiness (angle)
