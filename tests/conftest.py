# Import torch before any test module can pull in sklearn. On Windows,
# sklearn wheels bundle an older msvcp140.dll; if it loads first, torch's
# c10.dll fails to initialize (WinError 1114) for every later test module.
import torch  # noqa: F401
