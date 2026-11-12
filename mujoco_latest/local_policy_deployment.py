"""Reapply verified local policy deployments when rebuilding the runtime."""
import json
from pathlib import Path

RECEIPT=Path(__file__).resolve().parent/'policies/low_refine_20261111/deployment.json'
WHEEL_RECEIPT=Path(__file__).resolve().parent/'policies/wheel_speed250_20261112/deployment.json'
HANDSTAND_RECEIPT=Path(__file__).resolve().parent/'policies/handstand_distill_20261112/deployment.json'
GROUNDED_WHEEL_RECEIPT=Path(__file__).resolve().parent/'policies/wheel_groundedturn_20261112/deployment.json'

def apply_deployment(config):
    for receipt in (RECEIPT,WHEEL_RECEIPT,HANDSTAND_RECEIPT,GROUNDED_WHEEL_RECEIPT):
        if not receipt.exists():continue
        deployment=json.loads(receipt.read_text(encoding='utf-8'))
        for mode,spec in deployment['policies'].items():config['policies'][mode].update(spec)
        if 'stepping_profile' in deployment:config.update(deployment['stepping_profile'])
        if 'wheel_profile' in deployment:config['wheel_command_profile']=deployment['wheel_profile']
    return config
