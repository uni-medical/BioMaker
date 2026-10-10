"""Compatibility import; implementation lives in the installed BioMaker package."""
import sys
from biomaker.runtime import run_catalog as _implementation

sys.modules[__name__] = _implementation
