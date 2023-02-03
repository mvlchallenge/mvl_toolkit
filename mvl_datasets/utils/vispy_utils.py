from functools import partial

import vispy
from vispy.scene import Text, visuals


def setting_viewer(return_canvas=False, main_axis=True, bgcolor='black', caption=''):
    canvas = vispy.scene.SceneCanvas(keys='interactive',
                                     show=True,
                                     bgcolor=bgcolor)
    size_win = 1024
    canvas.size = 2*size_win, size_win

    t1 = Text(caption, parent=canvas.scene, color='white')
    t1.font_size = 24
    t1.pos = canvas.size[0] // 2, canvas.size[1] // 10
   
    view = canvas.central_widget.add_view()
    view.camera = 'arcball'  # turntable / arcball / fly / perspective
    
    if main_axis:
        visuals.XYZAxis(parent=view.scene)

    if return_canvas:
        return view, canvas
    return view


def setting_pcl(view, size=5, edge_width=2, antialias=0):
    scatter = visuals.Markers()
    scatter.set_gl_state('translucent',
                         depth_test=True,
                         blend=True,
                         blend_func=('src_alpha', 'one_minus_src_alpha'))
    # scatter.set_gl_state(depth_test=True)
    scatter.antialias = 0
    view.add(scatter)
    return partial(scatter.set_data, size=size, edge_width=edge_width)


def plot_color_plc(points,
                   color=(1, 1, 1, 1),
                   return_view=False,
                   size=0.5,
                   plot_main_axis=True,
                   background="black",
                   scale_factor=15,
                   caption=''
                   ):

    view = setting_viewer(main_axis=plot_main_axis, bgcolor=background, caption=caption)
    view.camera  = vispy.scene.TurntableCamera(elevation=45,
                                                      azimuth=45,
                                                      roll=0,
                                                      fov=0,
                                                      up='-y')
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

