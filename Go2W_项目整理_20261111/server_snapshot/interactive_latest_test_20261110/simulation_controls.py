"""Resume a paused Kit timeline on robot input, including inside a blocked physics step."""
import time


class SimulationControls:
    def __init__(self,controller,simulation,timeline,clock=time.monotonic):
        self.controller,self.simulation,self.timeline=controller,simulation,timeline
        self.clock=clock;self.resume_until=-1e9;self.resumes=0

    def update(self,event=None):
        c=self.controller;now=self.clock()
        with c.lock:
            if c.resume_requested:
                c.resume_requested=False
                # Kit shortcuts can execute after the robot's key callback.
                self.resume_until=now+2.
            playing=self.timeline.is_playing();stopped=self.timeline.is_stopped()
            if not playing and not stopped and now<=self.resume_until:
                # SimulationContext.play() pumps app.update(); invoking it from
                # an app update would recursively enter Kit. Use the timeline
                # here; PhysX's blocked wait_for_playing performs the Fabric sync.
                self.timeline.play();self.resumes+=1
                playing=self.timeline.is_playing()
                print('[SIMULATION_RESUMED] robot input resumed paused timeline',flush=True)
            c.simulation_state=dict(playing=playing,stopped=stopped,
                paused=not playing and not stopped,resumes=self.resumes,
                recovery='Press a robot command key or RESUME SIMULATION to resume pause')
