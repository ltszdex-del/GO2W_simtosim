"""Exercise actual MuJoCo mouse bindings without needing a desktop window."""
import unittest
from unittest.mock import patch
import bootstrap
import glfw
import mujoco
import numpy as np
from plain_viewer import Viewer

class CameraTests(unittest.TestCase):
    def setUp(self):
        self.viewer=Viewer.__new__(Viewer)
        self.viewer.model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
        self.viewer.scene=mujoco.MjvScene(self.viewer.model,maxgeom=10)
        self.viewer.cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.viewer.cam)
        self.viewer.cam.distance=2.6;self.viewer.window=None;self.viewer.last=(100,100)
    def test_scroll_both_directions(self):
        original=self.viewer.cam.distance
        self.viewer.scroll(None,0,1)
        changed=self.viewer.cam.distance
        self.assertNotEqual(original,changed)
        self.viewer.scroll(None,0,-1)
        self.assertTrue(np.isfinite(self.viewer.cam.distance))
        self.assertLess(abs(self.viewer.cam.distance-original),abs(changed-original))
    def test_drag(self):
        original=(self.viewer.cam.azimuth,self.viewer.cam.elevation)
        with patch.object(glfw,'get_mouse_button',return_value=glfw.PRESS),patch.object(glfw,'get_window_size',return_value=(1280,800)):
            self.viewer.motion(None,200,150)
        self.assertNotEqual(original,(self.viewer.cam.azimuth,self.viewer.cam.elevation))
    def test_minimized_window_drag(self):
        with patch.object(glfw,'get_mouse_button',return_value=glfw.PRESS),patch.object(glfw,'get_window_size',return_value=(0,0)):
            self.viewer.motion(None,101,101)
        self.assertTrue(np.isfinite(self.viewer.cam.azimuth))

if __name__=='__main__':
    print('Testing MuJoCo',mujoco.__version__,flush=True)
    unittest.main()
