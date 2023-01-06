import numpy as np
import torch


# import cv2
# import os


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def inverse_rigid_trans_4x4(Tr):
    """Inverse a rigid body transform matrix (4x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 4x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    inv_Tr[3, 3] = 1
    return inv_Tr


def inverse_rigid_trans_torch(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = torch.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = Tr[0:3, 0:3].transpose(0, 1)
    inv_Tr[0:3, 3] = torch.matmul(-inv_Tr[0:3, 0:3], Tr[0:3, 3])
    return inv_Tr


def homogeneous(xyz):
    """
    xyz: Nx3
    """
    return np.hstack((xyz, np.ones((xyz.shape[0], 1))))


def transform_xyz(xyz, Tr_matrix):
    """
    xyz: Nx3
    Tr_matrix: 3x4
    """
    assert isinstance(xyz, np.ndarray)
    assert isinstance(Tr_matrix, np.ndarray)
    hom_xyz = homogeneous(xyz)
    return np.dot(hom_xyz, Tr_matrix.T)
