import math

import numpy as np
from pyquaternion import Quaternion


def tum_pose2matrix44(l, seq="xyzw"):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    if seq == 'wxyz':
        if q[0] < 0:
            q *= -1
        q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    else:
        if q[3] < 0:
            q *= -1
        q = Quaternion(
            x=q[0],
            y=q[1],
            z=q[2],
            w=q[3],
        )
    transform = np.eye(4)
    transform[0:3, 0:3] = q.rotation_matrix
    transform[0:3, 3] = np.array(t)

    return transform


def isRotationMatrix(R):
    """
    Checks if a matrix is a valid rotation matrix.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def get_xyz_from_phi_coords(phi_coords):
    """
    Computes the xyz PCL from the ly_data (bearings_phi / phi_coords)
    """
    bearings_floor = get_bearings_from_phi_coords(
        phi_coords=phi_coords[1, :]
    )

    # ! Projecting bearing to 3D as pcl --> boundary
    # > Forcing ly-scale = 1
    ly_scale = 1 / bearings_floor[1, :]
    pcl_floor = ly_scale * bearings_floor
    return pcl_floor


def get_bearings_from_phi_coords(phi_coords):
    """
    Returns 3D bearing vectors (on the unite sphere) from phi_coords
    """
    W = phi_coords.__len__()
    u = np.linspace(0, W - 1, W)
    theta_coords = (2 * np.pi * u / W) - np.pi
    bearings_y = -np.sin(phi_coords)
    bearings_x = np.cos(phi_coords) * np.sin(theta_coords)
    bearings_z = np.cos(phi_coords) * np.cos(theta_coords)
    return np.vstack((bearings_x, bearings_y, bearings_z))


def stack_camera_poses(list_poses):
    """
    Stack a list of camera poses using Kronecker product
    https://en.wikipedia.org/wiki/Kronecker_product
    """
    M = np.zeros((list_poses.__len__()*3, list_poses.__len__()*4))
    for idx in range(list_poses.__len__()):
        aux = np.zeros((list_poses.__len__(), list_poses.__len__()))
        aux[idx, idx] = 1
        M += np.kron(aux, list_poses[idx][0:3, :])
    return M


def extend_array_to_homogeneous(array):
    """
    Returns the homogeneous form of a vector by attaching
    a unit vector as additional dimensions
    Parameters
    ----------
    array of (3, n) or (2, n)
    Returns (4, n) or (3, n)
    -------
    """
    try:
        assert array.shape[0] in (2, 3, 4)
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples))))

    except:
        assert array.shape[1] in (2, 3, 4)
        array = array.T
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples)))).T


def extend_vector_to_homogeneous_transf(vector):
    """
    Creates a homogeneous transformation (4, 4) given a vector R3
    :param vector: vector R3 (3, 1) or (4, 1)
    :return: Homogeneous transformation (4, 4)
    """
    T = np.eye(4)
    if vector.__class__.__name__ == "dict":
        T[0, 3] = vector["x"]
        T[1, 3] = vector["y"]
        T[2, 3] = vector["z"]
    elif type(vector) == np.array:
        T[0:3, 3] = vector[0:3, 0]
    else:
        T[0:3, 3] = vector[0:3]
    return T


def get_boundaries_from_corners(corners, cam_height):
    """
    Returns a set of points in 3D (3, n) for floor and ceiling boundaries using
    the ordered corners @corners and the camera_height (ceiling, floor)
    """
    # corners.append(corners[0])
    crn = extend_array_to_homogeneous(np.vstack(corners).T)[(0, 2, 1), :].T
    boundary = []
    for idx, c in enumerate(crn):
        v_dir = crn[(idx + 1) % corners.__len__(), :] - c
        wall_long = np.linalg.norm(v_dir)
        resolution = int(wall_long / 0.001)
        wall = c.reshape(3, 1) + v_dir.reshape(3, 1) * np.linspace(0, 1, resolution)

        boundary.append(wall)
    ceiling = np.hstack(boundary) * np.array((1, -cam_height[0], 1)).reshape(3, 1)
    floor = np.hstack(boundary) * np.array((1, cam_height[1], 1)).reshape(3, 1)
    return np.hstack((ceiling, floor))


def eulerAnglesToRotationMatrix(angles):
    theta = np.zeros((3))

    if angles.__class__.__name__ == 'dict':
        theta[0] = angles['x']
        theta[1] = angles['y']
        theta[2] = angles['z']
    else:
        theta[0] = angles[0]
        theta[1] = angles[1]
        theta[2] = angles[2]

    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def rotationMatrixToEulerAngles(R):
    """rotationMatrixToEulerAngles retuns the euler angles of a SO3 matrix
    """
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def vector2skew_matrix(vector):
    """
    Converts a vector [3,] into a matrix [3, 3] for cross product operations. v x v' = [v]v' where [v] is a skew representation of v
    :param vector: [3,]
    :return: skew matrix [3, 3]
    """
    vector = vector.ravel()
    assert vector.size == 3

    skew_matrix = np.zeros((3, 3))
    skew_matrix[1, 0] = vector[2]
    skew_matrix[2, 0] = -vector[1]
    skew_matrix[0, 1] = -vector[2]
    skew_matrix[2, 1] = vector[0]
    skew_matrix[0, 2] = vector[1]
    skew_matrix[1, 2] = -vector[0]

    return skew_matrix


def skew_matrix2vector(matrix):
    assert matrix.shape == (3, 3)

    vector = np.zeros((3, 1))

    vector[0] = matrix[2, 1]
    vector[1] = -matrix[2, 0]
    vector[2] = matrix[1, 0]

    return vector
