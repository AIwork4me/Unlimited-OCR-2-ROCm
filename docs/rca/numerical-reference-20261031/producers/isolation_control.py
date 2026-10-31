"""Experimental factor selection only. No arithmetic or global policy changes."""
def module_mode(condition, module):
    if module not in ('sam_model', 'vision_model'):
        raise ValueError('Unknown attention subsystem')
    parts = condition.split('/')
    if len(parts) != 2 or any(p not in ('MATH', 'AUTO') for p in parts):
        raise ValueError('Expected SAM/CLIP factor condition')
    return parts[0 if module == 'sam_model' else 1]

def dispatch_valid(dispatch, condition):
    if set(dispatch) != {'sam_model', 'vision_model'}:
        return False
    for module, ops in dispatch.items():
        expected = ('math' if module_mode(condition, module) == 'MATH' else
                    'efficient_attention' if module == 'sam_model' else 'flash_attention')
        attention = [op for op in ops if '_scaled_dot_product' in op]
        if not attention or any(expected not in op for op in attention):
            return False
    return True
