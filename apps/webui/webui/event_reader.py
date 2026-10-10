"""Compatibility import; implementation lives in the installed BioMaker package."""
import sys
from biomaker.runtime import event_reader as _implementation

sys.modules[__name__] = _implementation
