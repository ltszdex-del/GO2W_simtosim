"""Minimal GLFW viewer: robot keys never reach MuJoCo's built-in shortcuts."""
import contextlib
import glfw
import mujoco

# MuJoCo 3.3 takes (model, action, dx, dy, scene, camera); 3.17 removed
# the scene argument. Select the installed binding's documented signature.
_CAMERA_USES_SCENE = 'MjvScene' in (mujoco.mjv_moveCamera.__doc__ or '')

class Viewer:
    def __init__(self,model,data):
        if not glfw.init():raise RuntimeError('GLFW initialization failed')
        self.window=glfw.create_window(1280,800,'Go2W MuJoCo | LOW limits',None,None)
        if not self.window:raise RuntimeError('GLFW window creation failed')
        glfw.make_context_current(self.window);glfw.swap_interval(0)
        self.model,self.data=model,data
        self.cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(self.cam)
        self.opt=mujoco.MjvOption();mujoco.mjv_defaultOption(self.opt)
        self.user_scn=mujoco.MjvScene(model,maxgeom=32)
        self.scene=mujoco.MjvScene(model,maxgeom=10000)
        self.context=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_100)
        self.text=None;self.last=None
        glfw.set_scroll_callback(self.window,self.scroll)
        glfw.set_cursor_pos_callback(self.window,self.motion)
        # No MuJoCo Simulate key callback is installed.
    def scroll(self,window,x,y):
        self.move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y)
    def move_camera(self,action,dx,dy):
        if _CAMERA_USES_SCENE:
            mujoco.mjv_moveCamera(self.model,action,dx,dy,self.scene,self.cam)
        else:
            mujoco.mjv_moveCamera(self.model,action,dx,dy,self.cam)
    def motion(self,window,x,y):
        if self.last is not None and glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS:
            width,height=glfw.get_window_size(window)
            height=max(1,height)
            self.move_camera(mujoco.mjtMouse.mjMOUSE_ROTATE_V,(x-self.last[0])/height,(y-self.last[1])/height)
        self.last=(x,y)
    def lock(self):return contextlib.nullcontext()
    def set_texts(self,text):self.text=text
    def is_running(self):return not glfw.window_should_close(self.window)
    def sync(self):
        glfw.make_context_current(self.window)
        mujoco.mjv_updateScene(self.model,self.data,self.opt,None,self.cam,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        for i in range(self.user_scn.ngeom):
            source=self.user_scn.geoms[i];target=self.scene.geoms[self.scene.ngeom]
            mujoco.mjv_initGeom(target,source.type,source.size,source.pos,source.mat.ravel(),source.rgba)
            self.scene.ngeom+=1
        width,height=glfw.get_framebuffer_size(self.window);viewport=mujoco.MjrRect(0,0,width,height)
        mujoco.mjr_render(viewport,self.scene,self.context)
        if self.text:mujoco.mjr_overlay(self.text[0],self.text[1],viewport,self.text[2],self.text[3],self.context)
        glfw.swap_buffers(self.window);glfw.poll_events()
        if glfw.get_key(self.window,glfw.KEY_ESCAPE)==glfw.PRESS:glfw.set_window_should_close(self.window,True)
    def close(self):
        self.context.free();glfw.destroy_window(self.window);glfw.terminate()
