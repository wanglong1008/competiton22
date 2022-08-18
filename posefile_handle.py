import numpy as np

import angel_trans
import pymap3d as pm

b_lat = 36.383786
b_lon = 127.369974

f = open("gt/dataset5_all.txt", "r")
msgs = f.readlines()
poses = []
for msg in msgs:
    m = msg.split(" ")
    print(m)
    lon = float(m[1])
    lat = float(m[2])
    x, y, z = pm.geodetic2enu(lat, lon, 0, b_lat, b_lon, 0, deg=True)
    # x = x / 3 + 2
    # y = y / 3 + 2
    q = [m[4], m[5], m[6], m[7]]
    euler = angel_trans.quaternion2euler(q)
    rot = angel_trans.euler2rotation(euler)
    trans = np.array([x, y, 0]).reshape(3, 1)
    print(euler, rot)
    pose = np.concatenate((rot, trans), axis=1).reshape(1, 12)
    poses.append(pose)
poses = np.concatenate(poses, axis=0)
np.savetxt("gt/pose5.txt", poses, delimiter=' ', fmt='%1.8e')
