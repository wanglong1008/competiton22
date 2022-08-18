import numpy as np


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def load_poses_from_txt(file_name):
    """Load poses from txt (KITTI format)
    Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)
    Args:
        file_name (str): txt file path
    Returns:
        poses (dict): {idx: 4x4 array}
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row * 4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
    return poses


pose3 = np.loadtxt("gt/pose5.txt")[:, 3::4]
pose3 = np.transpose(pose3, (1, 0))
pose3_test = np.loadtxt("result/pose5_test.txt")[:, 3::4]
pose3_test = np.transpose(pose3_test, (1, 0))
r, t, scale = umeyama_alignment(pose3_test, pose3, True)

align_transformation = np.eye(4)
align_transformation[:3:, :3] = r
align_transformation[:3, 3] = t
print(r, t, scale)
print(1 / scale)

poses_gt = load_poses_from_txt("gt/pose5.txt")
poses_result = load_poses_from_txt("result/pose5_test.txt")
poses = []
for cnt in poses_result:
    poses_result[cnt][:3, 3] *= scale
    pose = align_transformation @ poses_result[cnt]
    poses.append(pose[:3, :].reshape(1, 12))

poses = np.concatenate(poses, axis=0)
# poses_result = np.array(poses_result)
np.savetxt("result/pose5_tes.txt", poses, delimiter=' ', fmt='%1.8e')

# evo_traj kitti result/pose1_test.txt --ref gt/pose1.txt -p --plot_mod=xy -as
** Please tell me who you are. Run git config --global user.email "925367077@qq.com" git config --global user.name "Your Name"
to set your account's default identity. Omit --global to set the identity only in this repository. unable to auto-detect email address (got '92536@wl_y9000p.(none)')