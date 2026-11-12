"""Explicit collision-only migration. Never accept changed inertials/joints/leg colliders."""
import copy
import hashlib
import xml.etree.ElementTree as ET

def _canonical(e):
    return (e.tag,tuple(sorted(e.attrib.items())),(e.text or '').strip(),tuple(_canonical(c) for c in e))

def collision_only_migration(original,current):
    a=ET.parse(original).getroot();b=ET.parse(current).getroot()
    changed=[]
    for name in ('base','Head_upper','Head_lower'):
        x=a.find(f"link[@name='{name}']");y=b.find(f"link[@name='{name}']")
        assert x is not None and y is not None,name
        if tuple(_canonical(v) for v in x.findall('collision'))!=tuple(_canonical(v) for v in y.findall('collision')):changed.append(name)
        for e in (x,y):
            for child in list(e.findall('collision')):e.remove(child)
    assert _canonical(a)==_canonical(b),'URDF differences extend beyond base/head collision geometry'
    return dict(kind='explicit_collision_geometry_migration',links=changed,
        original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        current_sha256=hashlib.sha256(current.read_bytes()).hexdigest())

