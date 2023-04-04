import mujoco as mj
from mujoco.glfw import glfw

import os
import numpy as np

class DARMRender():
    def __init__(self, model, data, window_size=(1200,900)):
        self.model = model
        self.data = data

        self.window_size = window_size # (width, height)

    def init_window_render(self):
        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(self.window_size[0], self.window_size[1], "DARM", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)   # TODO: What's this

        # Visualization
        self.cam = mj.MjvCamera()    # abstract camera
        self.opt = mj.MjvOption()    # visualization options
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        self.cam.azimuth = 90
        self.cam.elevation = -45
        self.cam.distance = 2
        self.cam.lookat = np.array([0.0, 0.0, 0])

        # For callback functions
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        def keyboard(window, key, scancode, act, mods):
            if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
                mj.mj_resetData(self.model, self.data)
                mj.mj_forward(self.model, self.data)

        def mouse_button(window, button, act, mods):
            # update button state
            self.button_left = (glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
            self.button_middle = (glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
            self.button_right = (glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

            # update mouse position
            glfw.get_cursor_pos(window) # TODO: Why is this needed again

        def mouse_move(window, xpos, ypos):
            # compute mouse displacement, save
            dx = xpos - self.lastx
            dy = ypos - self.lasty
            self.lastx = xpos
            self.lasty = ypos

            # no buttons down: nothing to do
            if (not self.button_left) and (not self.button_middle) and (not self.button_right):
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
            if self.button_right:
                if mod_shift:
                    action = mj.mjtMouse.mjMOUSE_MOVE_H
                else:
                    action = mj.mjtMouse.mjMOUSE_MOVE_V
            elif self.button_left:
                if mod_shift:
                    action = mj.mjtMouse.mjMOUSE_ROTATE_H
                else:
                    action = mj.mjtMouse.mjMOUSE_ROTATE_V
            else:
                action = mj.mjtMouse.mjMOUSE_ZOOM

            mj.mjv_moveCamera(self.model, action, dx/height,
                            dy/height, self.scene, self.cam)

        def scroll(window, xoffset, yoffset):
            action = mj.mjtMouse.mjMOUSE_ZOOM
            mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                            yoffset, self.scene, self.cam)

        glfw.set_key_callback(self.window, keyboard)
        glfw.set_cursor_pos_callback(self.window, mouse_move)
        glfw.set_mouse_button_callback(self.window, mouse_button)
        glfw.set_scroll_callback(self.window, scroll)

    def window_render(self, model=None, data=None):
        if not model: model = self.model
        if not data: data = self.data

        # Get Framebuffer Viewport
        vp_width, vp_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, vp_width, vp_height)

        # Update scene and render
        mj.mjv_updateScene(model, data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    def close_window(self):
        glfw.terminate()

if __name__ == "__main__":
    DARM_XML_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/mujoco_env/darm.xml"

    model = mj.MjModel.from_xml_path(DARM_XML_FILE)
    data = mj.MjData(model)
    
    print("Creating Darm Object...")
    darm = DARMRender(model, data)
    darm.init_window_render()
    while True:
        data.qpos[1] += 0.1
        mj.mj_forward(model, data)

        darm.window_render()