"""Labeled layout preview from the generated USD geometry."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
r=Path(__file__).resolve().parent;d=np.load(r/'preview_geometry.npz',allow_pickle=True);m=json.loads((r/'terrain_manifest.json').read_text())
fig,ax=plt.subplots(figsize=(14,11),facecolor='#eef1ef');ax.set_facecolor('#eef1ef')
polys=[];colors=[]
for v,faces,c in zip(d['vertices'],d['faces'],d['colors']):
 v=np.asarray(v,dtype=float)
 for f in faces:
  points=v[list(f)]
  normal=np.cross(points[1]-points[0],points[2]-points[0])
  if normal[2]>1e-10:
   polys.append(points[:,:2]);colors.append(c)
ax.add_collection(PolyCollection(polys,facecolors=colors,edgecolors=(.1,.15,.18,.5),linewidths=.45))
for obstacle in m['obstacles']:
 x,y=obstacle['origin_xy_m'];name=obstacle['name'].split('_')[0]
 if 'yaw_deg' in obstacle:
  a=np.deg2rad(obstacle['yaw_deg']);x+=np.cos(a)*obstacle['length_m']/2;y+=np.sin(a)*obstacle['length_m']/2
 ax.text(x,y,name+'\n'+str(round(obstacle['height_m']*100))+' cm',ha='center',va='center',fontsize=9,color='white',bbox=dict(boxstyle='round,pad=.25',facecolor='#17303a',alpha=.9,edgecolor='none'))
ax.text(0,0,'CLEAR RESET / CALIBRATION\n8 x 8 m',ha='center',va='center',color='white',fontsize=13,linespacing=1.6)
ax.set(xlim=(-19,19),ylim=(-15,15),xlabel='X / meters',ylabel='Y / meters');ax.set_aspect('equal');ax.grid(alpha=.1);ax.set_axisbelow(True)
ax.set_title('GO2W  /  SKATEPARK TRAINING TERRAIN\n36 x 28 m | 14 obstacle groups | meter scale, Z up',loc='left',fontsize=17,pad=18,color='#20333b')
fig.text(.11,.025,'S01-S07: stairs / terraces     R01-R04: banks / tabletop     Q01-Q02: curved ramps     L01: low ledge',fontsize=10,color='#394b52')
fig.savefig(r/'layout.png',dpi=150,bbox_inches='tight');plt.close(fig)
