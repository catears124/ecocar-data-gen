"""EcoCAR autocross simulation harness (CARLA + lightweight bicycle backend)."""
__version__ = "0.1.0"

import os as _os
import warnings as _warnings

_os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
_warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
