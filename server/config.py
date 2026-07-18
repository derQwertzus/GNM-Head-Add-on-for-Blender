"""Configuration for the GNM web wrapper.

Edit these if your model version/variant differs or you want a different port.
"""

# --- Server ---
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False  # set True while developing to get auto-reload + tracebacks

# --- Model selection ---
# These map to gnm_numpy.GNMMajorVersion / GNMVariant enum member names.
GNM_MAJOR_VERSION = "V3"   # e.g. "V3"
GNM_VARIANT = "HEAD"       # e.g. "HEAD"

# --- Texture ---
# Filename of the sample texture inside the package's data/textures folder.
# Leave as-is to use the shipped edge-flow map; point at any png/jpg you prefer.
SAMPLE_TEXTURE_NAME = "edgeflow_bw_4k.png"

# --- Display coloring (matches the original GNM demo viewer) ---
# Skin color in 0-255 sRGB. The demo's blue is (50, 156, 237).
# Scleras/teeth get brighter, irises darker, pupils black - like the demo.
SKIN_COLOR = (50, 156, 237)

# Saturation of the vertex colors: 1.0 = demo blue, 0.0 = fully gray clay.
VERTEX_COLOR_SATURATION = 0.0

# --- Slider ranges (defaults; tweak to taste) ---
IDENTITY_RANGE = (-3.0, 3.0)      # README typical range
EXPRESSION_RANGE = (-3.0, 3.0)
ROTATION_RANGE = (-1.5, 1.5)      # axis-angle radians
TRANSLATION_RANGE = (-0.3, 0.3)   # model units
