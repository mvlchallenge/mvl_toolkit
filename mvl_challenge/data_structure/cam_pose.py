import numpy as np
from mvl_challenge.utils.geometry_utils import isRotationMatrix


class CAM_REF:
    WC = "WC"  # ! World Coordinates
    CC = "CC"  # ! Cameras Coordinates
    WC_SO3 = "WC_SO3"  # ! World Coordinates only Rot applied
    ROOM = "ROOM_REF"  # ! Room Coordinates (primary frame)


class CamPose:
    def __init__(self, cfg):
        self.cfg = cfg
        self.SE3 = np.eye(4)
        self.vo_scale = 1
        self.gt_scale = 1
        self.idx = None

    @property
    def vo_scale(self):
        return self.__vo_scale

    @vo_scale.setter
    def vo_scale(self, value):
        assert value > 0
        self.__vo_scale = value

    @property
    def SE3(self):
        return self.__pose

    @SE3.setter
    def SE3(self, value):
        assert value.shape == (4, 4)
        self.__pose = value
        self.rot = value[0:3, 0:3]
        self.t = value[0:3, 3]

    @property
    def rot(self):
        return self.__rot

    @rot.setter
    def rot(self, value):
        assert isRotationMatrix(value)
        self.__rot = value
        self.__pose[:3, :3] = value

    @property
    def t(self):
        return self.__t * self.vo_scale * self.gt_scale

    @t.setter
    def t(self, value):
        assert value.reshape(3, ).shape == (3, )
        self.__t = value.reshape(3, )
        self.__pose[:3, 3] = value

    def SE3_scaled(self):
        m = np.eye(4)
        m[0:3, 0:3] = self.rot
        m[0:3, 3] = self.t
        return m
