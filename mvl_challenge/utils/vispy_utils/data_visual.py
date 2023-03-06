import numpy as np
from pyquaternion import Quaternion
from vispy import scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform, MatrixTransform
import vispy
from config import *
from mvl_challenge.data_structure import Layout
from mvl_challenge.utils.geometry_utils import extend_vector_to_homogeneous_transf


class CameraPoseVisual:
    def __init__(self, size=0.3, width=2, view=None):
        self.initial_pose = np.eye(4)
        self.width = width
        self.size = size
        # self.axis_z = scene.visuals.create_visual_node(visuals.ArrowVisual)
        self.base = np.zeros((4, 1))
        self.base[3, 0] = 1
        self.view = view
        self.isthereCam = False
        self.prev_pose = np.eye(4)

    def add_camera(self, pose, color, plot_sphere=True):
        if plot_sphere:
            self.sphere = scene.visuals.Sphere(
                radius=self.size * 0.5,
                # radius=1,
                method="latitude",
                parent=self.view.scene,
                color=color,
            )
        else:
            self.sphere = None

        self.initial_pose = pose
        pose_w = pose
        if self.sphere is not None:
            self.sphere.transform = STTransform(translate=pose_w[0:3, 3].T)
        # scene.visuals.Arrow
        self.isthereCam = True
        x = self.base + np.array([[self.size], [0], [0], [0]])
        y = self.base + np.array([[0], [self.size], [0], [0]])
        z = self.base + np.array([[0], [0], [self.size], [0]])

        pts = np.hstack([self.base, x, y, z])
        pts = np.dot(pose_w, pts)

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 3]
        self.axis_z = scene.Line(
            pos=pos, color=(0, 0, 1), method="gl", parent=self.view.scene
        )
        self.axis_z.set_data(pos=pos, color=(0, 0, 1))

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 1]
        self.axis_x = scene.Line(
            pos=pos, color=(1, 0, 0), method="gl", parent=self.view.scene
        )
        self.axis_x.set_data(pos=pos, color=(1, 0, 0))

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 2]
        self.axis_y = scene.Line(
            pos=pos, color=(0, 1, 0), method="gl", parent=self.view.scene
        )
        self.axis_y.set_data(pos=pos, color=(0, 1, 0))

        # self.axis_z(pos, width=self.width, color=(1, 1, 1), parent=self.view.scene)

    def transform_camera(self, pose):
        pose_w = pose @ np.linalg.inv(self.initial_pose)
        if self.sphere is not None:
            self.sphere.transform = STTransform(translate=pose[0:3, 3].T)
        q = Quaternion(matrix=pose_w[0:3, 0:3])
        trf = MatrixTransform()
        trf.rotate(angle=np.degrees(q.angle), axis=q.axis)
        trf.translate(pose_w[0:3, 3])
        self.axis_z.transform = trf
        self.axis_x.transform = trf
        self.axis_y.transform = trf


class DataVisual:
    def __init__(self):
        self.main_color = np.array((0, 255, 255)) / 255
        self.second_color = np.array((1, 0, 1))
        self.red_color = np.array((1, 0.1, 0.1))
        self.green_color = np.array((0, 255, 0)) / 255
        self.yellow_color = np.array((255, 255, 0)) / 255
        self.white_color = np.array((255, 255, 255)) / 255
        self.black_color = np.array((0, 0, 0))
        self.scatter_bounds = visuals.Markers()
        self.scatter_bounds.set_gl_state(
            "translucent",
            depth_test=True,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        # scatter.set_gl_state(depth_test=True)
        self.scatter_bounds.antialias = 0

        self.scatter_normals = visuals.Markers()
        self.scatter_normals.set_gl_state(
            "translucent",
            depth_test=True,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        # scatter.set_gl_state(depth_test=True)
        self.scatter_normals.antialias = 0

        self.scatter_position = visuals.Markers()
        self.scatter_position.set_gl_state(
            "translucent",
            depth_test=True,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        # scatter.set_gl_state(depth_test=True)
        self.scatter_position.antialias = 0

        self.scatter_planes = visuals.Markers()
        self.scatter_planes.set_gl_state(
            "translucent",
            depth_test=True,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        # scatter.set_gl_state(depth_test=True)
        self.scatter_planes.antialias = 0

        self.scatter_corners = visuals.Markers()
        self.scatter_corners.set_gl_state(
            "translucent",
            depth_test=True,
            blend=True,
            blend_func=("src_alpha", "one_minus_src_alpha"),
        )
        # scatter.set_gl_state(depth_test=True)
        self.scatter_corners.antialias = 0

        self.list_scatter = []
        self.cam = CameraPoseVisual()
        self.aux_cams = [CameraPoseVisual()]

    @staticmethod
    def set_cam_view(cam, view):
        cam.view = view

    def set_view(self, view):
        self.set_cam_view(self.cam, view)
        [self.set_cam_view(cam, view) for cam in self.aux_cams]

        view.add(self.scatter_bounds)
        view.add(self.scatter_corners)
        view.add(self.scatter_planes)
        view.add(self.scatter_normals)
        view.add(self.scatter_position)

    def plot_layout(self, ly: Layout, color):
        if ly.corner_coords is None:
            pts_corners = [ly.boundary[:, idx] for idx in ly.corners_id]
            pts_corners = np.vstack(pts_corners)
        else:
            pts_corners = ly.corner_coords

        if pts_corners.shape[0] == 3:
            pts_corners = pts_corners.T
        self.scatter_corners.set_data(
            pts_corners,
            edge_color=color.T,
            size=self.cfg.WIDTH_CORNER,
            edge_width=self.cfg.WIDTH_CORNER,
        )

        self.scatter_bounds.set_data(
            ly.boundary.T,
            edge_color=color.T,
            size=self.cfg.WIDTH_BOUNDARY,
            edge_width=self.cfg.WIDTH_BOUNDARY,
        )
        if not self.cam.isthereCam:
            self.cam.add_camera(ly.pose, color)
        else:
            self.cam.transform_camera(ly.pose)

    def plot_planes(self, list_planes, color):
        pcl = []
        cl = []
        for i, pl in enumerate(list_planes):
            local_pcl = []
            for s in np.linspace(0, 1, num=self.cfg.NUM_PTS_PER_PL // 2):
                local_pcl.append(pl.position - s * pl.line_dir)
                local_pcl.append(pl.position + s * pl.line_dir)
                # local_pcl.append(pl.distance * pl.normal.ravel())

            local_pcl = np.vstack(local_pcl)
            if color is None:
                c = np.ones_like(local_pcl) * self.main_color
            else:
                c = np.ones_like(local_pcl) * color

            pcl.append(local_pcl)
            cl.append(c)
            pose = pl.pose

        pcl = np.vstack(pcl)
        color = np.vstack(cl)

        if not self.cam.isthereCam:
            self.cam.add_camera(pose, self.second_color)
        else:
            self.cam.transform_camera(pose)

        self.scatter_planes.set_data(
            pcl,
            edge_color=color,
            size=self.cfg.THICK_PL,
            edge_width=self.cfg.THICK_PL / 2,
        )

    def plot_position_pl(self, list_pos_pl, color=None):
        pts = []
        color_pcl = []
        for p, i in zip(list_pos_pl, range(len(list_pos_pl))):
            if i < len(list_pos_pl) - 1:
                c = self.second_color
            else:
                if color is None:
                    c = self.main_color
                else:
                    c = color
            pts.append(p.position)
            color_pcl.append(np.ones_like(pts[-1]) * c)

        pcl = np.vstack(pts)
        color = np.vstack(color_pcl)
        self.scatter_position.set_data(
            pcl,
            edge_color=color,
            size=self.cfg.WIDTH_POSITION,
            edge_width=self.cfg.WIDTH_POSITION,
        )

    def plot_normals_pl(self, list_norm_pl, color):
        pts = []
        color_pcl = []
        for n, i in zip(list_norm_pl, range(len(list_norm_pl))):
            local_pcl = []
            if i < len(list_norm_pl) - 1:
                c = self.second_color
                size = self.cfg.WIDTH_NORMAL / 5
                number_pts = self.cfg.NUM_PTS_PER_PL // 4
            else:
                if color is None:
                    c = self.main_color
                else:
                    c = color

                size = self.cfg.WIDTH_NORMAL
                number_pts = self.cfg.NUM_PTS_PER_PL
            for norm_vect in n:
                for s in np.linspace(0, 1, num=number_pts):
                    local_pcl.append(0.1 * norm_vect + s * norm_vect)

            pts.append(np.vstack(local_pcl))
            color_pcl.append(np.ones_like(pts[-1]) * c)

        pcl = np.vstack(pts) * 3
        color = np.vstack(color_pcl)
        self.scatter_normals.set_data(pcl, edge_color=color, size=size, edge_width=size)

    def plot_layout_and_planes(self, ly: Layout, list_planes, color=None):
        self.plot_layout(ly, self.second_color)
        if color is None:
            self.plot_planes(list_planes, self.main_color)
        else:
            self.plot_planes(list_planes, color)

    def plot_aux_ref(self, pose, size, idx=0):
        camera_pose = extend_vector_to_homogeneous_transf(pose)
        if not self.aux_cams[idx].isthereCam:
            self.aux_cams[idx].size = size
            self.aux_cams[idx].add_camera(camera_pose, self.main_color)
        else:
            self.aux_cams[idx].transform_camera(camera_pose)

    def plot_pcl(self, pcl, color, size=None, camera_pose=None):
        if pcl.shape[0] == 3:
            pcl = pcl.T
            color = color.T

        assert pcl.shape[1] == 3

        if size is None:
            size = self.cfg.WIDTH_BOUNDARY

        if camera_pose is not None:
            if not self.cam.isthereCam:
                self.cam.add_camera(camera_pose, color)
            else:
                self.cam.transform_camera(camera_pose)

        self.scatter_bounds.set_data(pcl, edge_color=color, size=size, edge_width=size)


class Visualization:
    def __init__(self, caption=""):
        # ! Setting up vispy
        self.canvas = vispy.scene.SceneCanvas(keys="interactive", bgcolor="white")
        res = 1024 * 2
        self.canvas.size = res, res // 2
        self.canvas.show()

        # Create two ViewBoxes, place side-by-side
        self.vb1 = scene.widgets.ViewBox(parent=self.canvas.scene)
        self.vb2 = scene.widgets.ViewBox(parent=self.canvas.scene)

        # visuals.XYZAxis(parent=self.vb1.scene)
        grid = self.canvas.central_widget.add_grid()
        grid.padding = 6
        grid.add_widget(self.vb1, 0, 0)
        grid.add_widget(self.vb2, 0, 1)

        self.vb1.camera = vispy.scene.TurntableCamera(
            elevation=90, azimuth=90, roll=0, fov=0, up="-y"
        )

        self.vb2.camera = vispy.scene.TurntableCamera(
            elevation=90, azimuth=90, roll=0, fov=0, up="-y"
        )

        self.vb1.camera.link(self.vb2.camera)

        # self.vb1.camera = vispy.scene.TurntableCamera(elevation=25,
        #                                               azimuth=-10,
        #                                               roll=0,
        #                                               fov=0,
        #                                               up='-y')
        self.vb1.camera.scale_factor = 25

        self.cam_visual = DataVisual()
        self.layout_visual = DataVisual()
        self.pcl_visual = DataVisual()

        self.cam_visual_v2 = DataVisual()
        self.layout_visual_v2 = DataVisual()
        self.pcl_visual_v2 = DataVisual()

        self.cam_visual.set_view(self.vb1)
        self.layout_visual.set_view(self.vb1)
        self.pcl_visual.set_view(self.vb1)

        self.cam_visual_v2.set_view(self.vb2)
        self.layout_visual_v2.set_view(self.vb2)
        self.pcl_visual_v2.set_view(self.vb2)

    def set_visual_list(self, len):
        self.list_visuals = []
        [self.list_visuals.append(DataVisual()) for _ in range(len)]
        [lv.set_view(self.vb1) for lv in self.list_visuals]
