"""Continuous reference-driven idle recovery, independent of residual motion."""
import math

def recovery_goal(keys,command):
    if any(k in keys for k in 'WSQE'):return 0.
    return math.exp(-(math.hypot(*command[:2])/.12)**2-(command[2]/.18)**2)

def blend_step(value,goal,dt):
    return value+max(-dt/.2,min(dt/.8,goal-value))
