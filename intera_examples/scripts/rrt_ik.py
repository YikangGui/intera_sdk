#! /usr/bin/env python
# Copyright (c) 2013-2018, Rethink Robotics Inc.
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

"""
Intera RSDK Inverse Kinematics Example
"""
import rospy
import argparse
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_interface import Limb

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

def call_ik_service(x, y, seed_position = 'home', limb_name = "right"):
    limb = Limb()
    ns = "ExternalTools/" + limb_name + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
        'right': PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=x,
                    y=y,
                    z=0.15,
                ),
                orientation=Quaternion(
                    x=0.0,
                    y=1.0,
                    z=0.0,
                    w=0.0,
                ),
            ),
        ),
    }
    # Add desired pose for inverse kinematics
    ikreq.pose_stamp.append(poses[limb_name])
    # Request inverse kinematics from base to "right_hand" link
    ikreq.tip_names.append('right_hand')

    # Optional Advanced IK parameters
    rospy.loginfo("Running Advanced IK Service Client.")
    # The joint seed is where the IK position solver starts its optimization
    ikreq.seed_mode = ikreq.SEED_USER
    seed = JointState()
    seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                    'right_j4', 'right_j5', 'right_j6']
    if seed_position == 'home':
        seed.position = [0.00, -0.53, -0.46, 1.29, 0.51, 0.94, 0] # home pos
    elif seed_position == 'current':
        seed.position = limb.joint_ordered_angles()
    else:
        seed.position = seed_position
    ikreq.seed_angles.append(seed)

    # Once the primary IK task is solved, the solver will then try to bias the
    # the joint angles toward the goal joint configuration. The null space is 
    # the extra degrees of freedom the joints can move without affecting the
    # primary IK task.
    ikreq.use_nullspace_goal.append(True)
    # The nullspace goal can either be the full set or subset of joint angles
    goal = JointState()
    goal.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                    'right_j4', 'right_j5', 'right_j6']
    goal.position = [0.00, -0.53, -0.46, 1.29, 0.51, 0.94, 0]
    ikreq.nullspace_goal.append(goal)
    # The gain used to bias toward the nullspace goal. Must be [0.0, 1.0]
    # If empty, the default gain of 0.4 will be used
    ikreq.nullspace_gain.append(0.4)

    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException) as e:
        rospy.logerr("Service call failed: %s" % (e,))
        return False, None

    # Check if result valid, and type of seed ultimately used to get solution
    if (resp.result_type[0] > 0):
        seed_str = {
                    ikreq.SEED_USER: 'User Provided Seed',
                    ikreq.SEED_CURRENT: 'Current Joint Angles',
                    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                   }.get(resp.result_type[0], 'None')
        rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
              (seed_str,))
        # Format solution into Limb API-compatible dictionary
        limb_joints = dict(list(zip(resp.joints[0].name, resp.joints[0].position)))
        rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
        rospy.loginfo("------------------")
        rospy.loginfo("Response Message:\n%s", resp)
    else:
        rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
        rospy.logerr("Result Error %d", resp.result_type[0])
        return False, None
    
    for i in limb.joint_ordered_angles():
        print('%.2f' % i, end=' ')
    print()

    for i in resp.joints[0].position:
        print('%.2f' % i, end=' ')
    print()

    return True, resp.joints[0].position


def main():
    """RSDK Inverse Kinematics Example

    A simple example of using the Rethink Inverse Kinematics
    Service which returns the joint angles and validity for
    a requested Cartesian Pose.

    Run this example, the example will use the default limb
    and call the Service with a sample Cartesian
    Pose, pre-defined in the example code, printing the
    response of whether a valid joint solution was found,
    and if so, the corresponding joint angles.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    parser.add_argument(
        "-p", "--position", type=float,
        nargs='+',
        help="Desired end position: X, Y")
    
    args = parser.parse_args(rospy.myargv()[1:])
    
    x = args.position[0]
    y = args.position[1]

    rospy.init_node("rsdk_ik_service_client")

    if call_ik_service(x, y)[0]:
        rospy.loginfo("Advanced IK call passed!")
    else:
        rospy.logerr("Advanced IK call FAILED")


if __name__ == '__main__':
    main()
