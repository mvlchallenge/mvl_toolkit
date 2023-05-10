import numpy as np
import vispy
from functools import partial
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from vispy.scene.visuals import Text
from vispy import app, scene, io
from .data_visual import DataVisual, Visualization
import sys
from matplotlib.colors import hsv_to_rgb
import vispy.io as vispy_file
import os


def plot_list_ly(list_ly=None, ceil_or_floor="", save_at=None):
    room_scene = list_ly[0].cfg._room_scene
    caption = room_scene
    if list_ly.__len__() < 1:
        return

    # ! Setting up vispy
    canvas = vispy.scene.SceneCanvas(
        keys="interactive", bgcolor="white", create_native=True
    )
    res = 1024 * 2
    canvas.size = res, res // 2
    canvas.show()

    t1 = Text(room_scene, parent=canvas.scene, color="black")
    t1.font_size = 24
    t1.pos = canvas.size[0] // 2, canvas.size[1] // 10

    # Create two ViewBoxes, place side-by-side
    vb1 = scene.widgets.ViewBox(parent=canvas.scene)
    visuals.XYZAxis(parent=vb1.scene)
    grid = canvas.central_widget.add_grid()
    grid.padding = 6
    grid.add_widget(vb1, 0, 0)
    vb1.camera = vispy.scene.TurntableCamera(
        elevation=90, azimuth=180, roll=0, fov=0, up="-y"
    )

    raw_ly_visual = DataVisual()
    ly_visual = DataVisual()

    raw_ly_visual.set_view(vb1)
    ly_visual.set_view(vb1)

    raw = []
    # [raw.append(np.linalg.inv(ly.pose)[0:3, :] @ extend_array_to_homogeneous(ly.boundary)) for ly in list_ly]
    # list_pl = flatten_lists_of_lists([ly.list_planes for ly in list_obj])
    if ceil_or_floor == "":
        [raw.append(ly.boundary_floor) for ly in list_ly]
        [raw.append(ly.boundary_ceiling) for ly in list_ly]
    elif ceil_or_floor == "ceil":
        [raw.append(ly.boundary_ceiling) for ly in list_ly]
    elif ceil_or_floor == "floor":
        [raw.append(ly.boundary_floor) for ly in list_ly]

    pcl_raw = np.hstack(raw)
    max_size = np.max(
        pcl_raw - np.mean(pcl_raw, axis=1, keepdims=True), axis=1, keepdims=True
    )

    vb1.camera.scale_factor = np.max(max_size) * 2.5
    vb1.camera.center = np.mean(pcl_raw, axis=1, keepdims=True)

    # print(max_size)
    raw_ly_visual.plot_pcl(pcl_raw, raw_ly_visual.black_color, size=0.5)

    if save_at is not None:
        os.makedirs(save_at, exist_ok=True)
        res = canvas.render()
        vispy_file.write_png(os.path.join(save_at, f"{caption}.png"), res)
        return
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()


def plot_list_pcl(list_pcl, sizes, caption="Plotting PCL"):
    vis = Visualization(caption=caption)
    vis.set_visual_list(list_pcl.__len__())

    assert list_pcl.__len__() == sizes.__len__()
    np.random.seed(500)
    color_idx = (0, vis.colors_list.shape[1], list_pcl.__len__())

    for i, visual, pcl, size in zip(
        range(sizes.__len__()), vis.list_visuals, list_pcl, sizes
    ):
        visual.plot_pcl(pcl, vis.colors_list[:, i], size)

    vis.canvas.show()
    if sys.flags.interactive == 0:
        app.run()


def get_color_list(array_colors=None, fr=0.1, return_list=False, number_of_colors=None):
    """
    Returns a different color RGB for every element in the array_color
    """
    if array_colors is not None:
        number_of_colors = len(array_colors)

    h = np.linspace(0.1, 0.8, number_of_colors)
    np.random.shuffle(h)
    # values = np.linspace(0, np.pi, number_of_colors)
    colors = np.ones((3, number_of_colors))

    colors[0, :] = h

    return hsv_to_rgb(colors.T).T


def setting_viewer(return_canvas=False, main_axis=True, bgcolor="black", caption=""):
    canvas = vispy.scene.SceneCanvas(keys="interactive", show=True, bgcolor=bgcolor)
    size_win = 1024
    canvas.size = 2 * size_win, size_win

    t1 = Text(caption, parent=canvas.scene, color="white")
    t1.font_size = 24
    t1.pos = canvas.size[0] // 2, canvas.size[1] // 10

    view = canvas.central_widget.add_view()
    view.camera = "arcball"  # turntable / arcball / fly / perspective

    if main_axis:
        visuals.XYZAxis(parent=view.scene)

    if return_canvas:
        return view, canvas
    return view


def setting_pcl(view, size=5, edge_width=2, antialias=0):
    scatter = visuals.Markers()
    scatter.set_gl_state(
        "translucent",
        depth_test=True,
        blend=True,
        blend_func=("src_alpha", "one_minus_src_alpha"),
    )
    # scatter.set_gl_state(depth_test=True)
    scatter.antialias = 0
    view.add(scatter)
    return partial(scatter.set_data, size=size, edge_width=edge_width)


def plot_color_plc(
    points,
    color=(0, 0, 0, 1),
    return_view=False,
    size=0.5,
    plot_main_axis=True,
    background="white",
    scale_factor=15,
    caption="",
):

    view = setting_viewer(main_axis=plot_main_axis, bgcolor=background, caption=caption)
    view.camera = vispy.scene.TurntableCamera(
        elevation=90, azimuth=90, roll=0, fov=0, up="-y"
    )
    # view.camera = vispy.scene.TurntableCamera(elevation=90,
    #                                           azimuth=0,
    #                                           roll=0,
    #                                           fov=0,
    #                                           up='-y')
    view.camera.scale_factor = scale_factor
    draw_pcl = setting_pcl(view=view)
    draw_pcl(points, edge_color=color, size=size)
    if not return_view:
        vispy.app.run()
    else:
        return view
