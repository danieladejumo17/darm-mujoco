import os
import numpy as np

import mujoco as mj
from mujoco.glfw import glfw


# XML Path
xml_path = "../mujoco_env/darm.xml"


# Simulation Configs
simend = 100    # simulation duration
print_camera_config = False
window_size = 1200, 900 # width, height


# Load Model
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)


# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(window_size[0], window_size[1], "DARM", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)   # TODO: What's this

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window) # TODO: Why is this needed again

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)    # TODO: Look into this, height/width issue

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)


# Visualization
cam = mj.MjvCamera()    # abstract camera
opt = mj.MjvOption()    # visualization options
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cam.azimuth = 90
cam.elevation = -45
cam.distance = 2
cam.lookat = np.array([0.0, 0.0, 0])


# Controller
def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    
    # TODO: Create dict of joint no by names
    idx = mj.mj_name2id(model, int(mj.mjtObj.mjOBJ_ACTUATOR), "flexor_superficialis_ii_actuator")
    print(mj.mj_id2name(model, int(mj.mjtObj.mjOBJ_ACTUATOR), 2))
    print(idx)

    data.ctrl[idx] = 2
    data.ctrl[2] = 20


init_controller(model,data)
mj.set_mjcb_control(controller)


# Simulation Loop
while not glfw.window_should_close(window):
    time_prev = data.time   # simulation time in seconds

    # Perform loop tasks at 60Hz. Once every 1/60 seconds
    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if data.time > simend:
        break

    # Print camera configuration
    if print_camera_config:
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Get Framebuffer Viewport
    vp_width, vp_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, vp_width, vp_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()