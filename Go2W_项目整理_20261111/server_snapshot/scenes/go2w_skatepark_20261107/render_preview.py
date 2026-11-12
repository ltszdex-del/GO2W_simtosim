"""Render the actual USD in the installed Isaac Lab 3 / Isaac Sim runtime."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import omni.usd
 import omni.kit.app
 enabled=omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.replicator.core',True)
 assert enabled,'Replicator extension unavailable'
 import omni.replicator.core as rep
 from PIL import Image
 r=Path(__file__).resolve().parent
 omni.usd.get_context().open_stage(str(r/'preview.usda'))
 for _ in range(30):app.update()
 product=rep.create.render_product('/World/Camera',(1600,1000))
 annot=rep.AnnotatorRegistry.get_annotator('rgb');annot.attach(product)
 # This runtime renders attached products on app updates; no timeline capture is needed.
 for _ in range(100):app.update()
 data=annot.get_data();assert data.size>0
 Image.fromarray(data).save(r/'preview.png');print('RENDERED',data.shape,flush=True)
except BaseException:
 import traceback;traceback.print_exc();app.close(exit_code=1)
else:app.close()
