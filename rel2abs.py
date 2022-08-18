import numpy as np
import angel_trans
import pymap3d as pm

global_pose = np.eye(4)
poses = []
global_lat = 1
global_lon = 1
global_alt = 0


def rel2abs_pose(scale, euler, trans):
    global global_pose, poses
    rot = angel_trans.euler2rotation(euler)
    quaternion = angel_trans.euler2quaternion(euler)
    trans = np.multiply(scale, np.array(trans))
    trans = trans.reshape(3, 1)
    pose = np.concatenate((rot, trans), axis=1)
    pose_mat = np.vstack([pose, np.array([0, 0, 0, 1])])
    global_pose = global_pose @ np.linalg.inv(pose_mat)
    poses.append(global_pose[0:3, :].reshape(1, 12))
    lat, lon, alt = pm.enu2geodetic(trans[0], trans[1], trans[2], global_lat, global_lon, global_alt)
    return lat, lon, quaternion


print(rel2abs_pose(2, [0, 0, 0], [1, 2, 3]))
print(global_pose)
rel2abs_pose(2, [0, 0, 0], [1, 2, 3])
print(global_pose)
