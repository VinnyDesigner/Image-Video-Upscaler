import os
import sys
import traceback
import torch
import warnings

# Robust suppression for the persistent torch.load FutureWarning
# We monkey-patch torch.load to explicitly set weights_only=False (the current default)
# and wrap the call in a warning suppression block.
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
warnings.filterwarnings("ignore", category=FutureWarning)

# Windows Fix: Resolve DLL initialization issues (WinError 1114)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Compatibility Fix: Inject missing torchvision module into sys.modules
try:
    import torchvision.transforms.functional as F
    import types
    # Create a dummy module
    mock_module = types.ModuleType("torchvision.transforms.functional_tensor")
    # Copy all functions from F to the mock module
    for name in dir(F):
        setattr(mock_module, name, getattr(F, name))
    # Inject it into sys.modules so 'import torchvision.transforms.functional_tensor' works
    sys.modules["torchvision.transforms.functional_tensor"] = mock_module
except Exception:
    pass

try:
    from ui import demo
except Exception as e:
    print(f"\nFATAL ERROR: Failed to initialize application.")
    traceback.print_exc() # Show exactly where it's failing
    sys.exit(1)

if __name__ == "__main__":
    print("Starting AI Video Upscaler Pro...")
    demo.launch()
