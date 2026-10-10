"""Compatibility import; implementation lives in the installed BioMaker package."""
import sys
from biomaker.runtime import runner as _implementation

if __name__ == "__main__":
    import runpy
    runpy.run_module("biomaker.runtime.runner", run_name="__main__")
else:
    sys.modules[__name__] = _implementation
