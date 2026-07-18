"""Local Flask server that wraps the GNM head model for a Three.js frontend.

Run from inside the same virtual environment that has `gnm` installed:

    pip install flask
    python server.py

Then open http://127.0.0.1:5000 in your browser.

Static geometry (faces + UVs) is sent once via /api/meta. Only the vertex
positions change per parameter update, so /api/mesh returns just those,
base64-encoded as raw float32 for speed.
"""

import base64
import logging
import os

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

import config

# Keep the log readable: werkzeug prints one line per request otherwise.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model():
    """Import GNM and load the configured version/variant once."""
    from gnm.shape import gnm_numpy  # imported here for a clear error if missing

    version = getattr(gnm_numpy.GNMMajorVersion, config.GNM_MAJOR_VERSION)
    variant = getattr(gnm_numpy.GNMVariant, config.GNM_VARIANT)
    model = gnm_numpy.GNM.from_local(version=version, variant=variant)
    return model, gnm_numpy


print("Loading GNM model ...")
GNM, GNM_NUMPY = load_model()

# Static geometry pulled once.
FACES = np.asarray(GNM.triangles, dtype=np.int32).reshape(-1, 3)
NUM_VERTICES = int(GNM.num_vertices)
NUM_JOINTS = int(GNM.num_joints)
IDENTITY_DIM = int(GNM.identity_dim)
EXPRESSION_DIM = int(GNM.expression_dim)

# Per-vertex UVs (may be lossy across seams) and per-corner UVs (seam-correct).
VERTEX_UVS = np.asarray(GNM.vertex_uvs, dtype=np.float32).reshape(-1, 2)
TRIANGLE_UVS = np.asarray(GNM.triangle_uvs, dtype=np.float32).reshape(-1, 2)

# Native quad topology (for DCC clients — clean edge loops).
try:
    QUADS = np.asarray(GNM.quads, dtype=np.int32).reshape(-1, 4)
    QUAD_UVS = np.asarray(GNM.quad_uvs, dtype=np.float32).reshape(-1, 2)
except Exception as _e:
    print(f"  (quad topology unavailable: {_e})")
    QUADS = None
    QUAD_UVS = None


# --- Display coloring + display faces, replicating the original demo viewer ---
# (gnm_jupyter_viewer + visualization/vertex_colors: skin color scaled/offset
#  per region, pupils black, eye exterior shells excluded from rendering.)

_COLOR_MODIFIERS = {
    "skin": (1.0, 0.0),
    "scleras": (0.6, 0.4),
    "irises": (0.6, 0.0),
    "gums": (0.7, 0.0),
    "teeth": (0.6, 0.4),
    "tongue": (0.7, 0.0),
    "mouth_sock": (0.7, 0.0),
    "eye_exteriors": (0.6, 0.4),  # cornea shells: sclera-bright, never black
}


def _srgb_to_linear(c):
    """Modern three.js treats vertex colors as linear; demo colors are sRGB."""
    return np.power(np.clip(c, 0.0, 1.0), 2.2)


def _build_vertex_colors():
    base = np.array([c / 255.0 for c in config.SKIN_COLOR], dtype=np.float64)
    # Start from skin everywhere: vertex groups not in the modifier table
    # (e.g. caruncles) inherit skin instead of rendering black.
    colors = np.tile(base, (NUM_VERTICES, 1))
    try:
        names = list(GNM.vertex_group_names)
        for region, (scale, offset) in _COLOR_MODIFIERS.items():
            if region in names:
                colors[GNM.vertex_group_indices(region)] = base * scale + offset
        if "pupils" in names:
            colors[GNM.vertex_group_indices("pupils")] = 0.0
    except Exception as e:
        print(f"  (vertex group coloring failed, flat color: {e})")
        colors[:] = base
    # Desaturate toward luminance (config; pupils stay black either way).
    sat = float(getattr(config, "VERTEX_COLOR_SATURATION", 1.0))
    lum = colors @ np.array([0.2126, 0.7152, 0.0722])
    colors = lum[:, None] + (colors - lum[:, None]) * sat
    return _srgb_to_linear(colors).astype(np.float32)


def _build_display_faces():
    """Demo renders triangles_group('~eye_exteriors') to hide cornea shells."""
    try:
        return np.asarray(
            GNM.triangles_group("~eye_exteriors"), dtype=np.int32
        ).reshape(-1, 3)
    except Exception as e:
        print(f"  (triangles_group unavailable, showing all faces: {e})")
        return FACES


VERTEX_COLORS = _build_vertex_colors()
DISPLAY_FACES = _build_display_faces()


# Per-face component ids for show/hide toggles in DCC clients:
# 0 = skin/other, 1 = teeth (incl. gums), 2 = tongue, 3 = eyes.
_COMPONENT_SELECTORS = {
    1: ("teeth", "gums"),
    2: ("tongue",),
    3: ("scleras", "irises", "pupils"),
    4: ("eye_exteriors",),   # transparent cornea shells — occlude in solid view
}


def _component_vertex_sets():
    sets = {}
    for cid, names in _COMPONENT_SELECTORS.items():
        verts = []
        for name in names:
            try:
                verts.append(np.asarray(GNM.vertex_group_indices(name)).reshape(-1))
            except Exception:
                pass  # group not present in this model version
        if verts:
            sets[cid] = np.unique(np.concatenate(verts))
    return sets


def _classify_faces(face_array):
    """Vertex-membership classification: a face belongs to a component when
    ALL its vertices are in that component's vertex-group union — this catches
    boundary faces spanning sub-groups (e.g. the pupil-iris ring)."""
    comp = np.zeros(face_array.shape[0], dtype=np.int32)
    try:
        for cid, vset in _component_vertex_sets().items():
            comp[np.isin(face_array, vset).all(axis=1)] = cid
    except Exception as e:
        print(f"  (component classification failed: {e})")
    return comp


FACE_COMPONENTS = _classify_faces(FACES)
QUAD_COMPONENTS = _classify_faces(QUADS) if QUADS is not None else None

# Path to the sample texture inside the installed package.
_PKG_DIR = os.path.dirname(GNM_NUMPY.__file__)
# gnm_numpy lives in .../gnm/shape, textures live in .../gnm/shape/data/textures
_SHAPE_DIR = _PKG_DIR
TEXTURE_PATH = os.path.join(_SHAPE_DIR, "data", "textures", config.SAMPLE_TEXTURE_NAME)

print(
    f"Loaded: {NUM_VERTICES} verts, {FACES.shape[0]} faces, "
    f"identity_dim={IDENTITY_DIM}, expression_dim={EXPRESSION_DIM}, "
    f"joints={NUM_JOINTS}"
)
if not os.path.exists(TEXTURE_PATH):
    print(f"  (note: sample texture not found at {TEXTURE_PATH})")


# --------------------------------------------------------------------------
# Parameter group metadata
# --------------------------------------------------------------------------

def _clip_groups(groups, dim):
    """Clamp (label, start, end) group ranges to the actual dimension."""
    out = []
    for label, start, end in groups:
        start = max(0, min(start, dim))
        end = max(start, min(end, dim))
        if end > start:
            out.append({"label": label, "start": start, "end": end})
    return out


# Component breakdown from the GNM README (v3.x). Clamped to real dims so a
# different model version still produces sane groups.
IDENTITY_GROUPS = _clip_groups(
    [("head", 0, 170), ("eyeball", 170, 173), ("teeth", 173, 253)],
    IDENTITY_DIM,
)
EXPRESSION_GROUPS = _clip_groups(
    [
        ("left eye", 0, 100),
        ("right eye", 100, 200),
        ("lower face", 200, 350),
        ("tongue", 350, 382),
        ("iris", 382, 383),
    ],
    EXPRESSION_DIM,
)


def _names(attr, count, prefix):
    """Return model-provided names if present and correctly sized, else fallback."""
    vals = getattr(GNM, attr, None)
    if vals is not None:
        try:
            vals = [str(v) for v in vals]
            if len(vals) == count:
                return vals
        except TypeError:
            pass
    return [f"{prefix}{i}" for i in range(count)]


JOINT_NAMES = _names("joint_names", NUM_JOINTS, "joint")
IDENTITY_NAMES = _names("identity_names", IDENTITY_DIM, "i")
EXPRESSION_NAMES = _names("expression_names", EXPRESSION_DIM, "e")


# --------------------------------------------------------------------------
# Encoding helpers
# --------------------------------------------------------------------------

def b64_f32(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def b64_i32(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.int32).tobytes()).decode("ascii")


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/api/meta")
def api_meta():
    """Static model info: dims, groups, names, faces, UVs, texture availability."""
    return jsonify({
        "numVertices": NUM_VERTICES,
        "numFaces": int(FACES.shape[0]),
        "numJoints": NUM_JOINTS,
        "identityDim": IDENTITY_DIM,
        "expressionDim": EXPRESSION_DIM,
        "identityGroups": IDENTITY_GROUPS,
        "expressionGroups": EXPRESSION_GROUPS,
        "jointNames": JOINT_NAMES,
        "identityNames": IDENTITY_NAMES,
        "expressionNames": EXPRESSION_NAMES,
        "ranges": {
            "identity": list(config.IDENTITY_RANGE),
            "expression": list(config.EXPRESSION_RANGE),
            "rotation": list(config.ROTATION_RANGE),
            "translation": list(config.TRANSLATION_RANGE),
        },
        "hasTexture": os.path.exists(TEXTURE_PATH),
        # base64 raw binary — decoded client-side into typed arrays
        "facesB64": b64_i32(FACES.reshape(-1)),       # full mesh, for export
        "displayFacesB64": b64_i32(DISPLAY_FACES.reshape(-1)),  # eye shells hidden
        "faceComponentsB64": b64_i32(FACE_COMPONENTS),  # 0 skin, 1 teeth, 2 tongue, 3 eyes, 4 eye covers
        **({"numQuads": int(QUADS.shape[0]),
            "quadsB64": b64_i32(QUADS.reshape(-1)),
            "quadUvsB64": b64_f32(QUAD_UVS.reshape(-1)),
            "quadComponentsB64": b64_i32(QUAD_COMPONENTS)}
           if QUADS is not None else {}),
        "vertexColorsB64": b64_f32(VERTEX_COLORS.reshape(-1)),  # numVertices*3, linear
        "vertexUvsB64": b64_f32(VERTEX_UVS.reshape(-1)),      # numVertices*2
        "triangleUvsB64": b64_f32(TRIANGLE_UVS.reshape(-1)),  # numFaces*3*2
    })


@app.route("/api/mesh", methods=["POST"])
def api_mesh():
    """Evaluate GNM with the supplied parameters, return vertex positions."""
    data = request.get_json(force=True)

    identity = np.asarray(data.get("identity", []), dtype=np.float32)
    expression = np.asarray(data.get("expression", []), dtype=np.float32)
    rotations = np.asarray(data.get("rotations", []), dtype=np.float32)
    translation = np.asarray(data.get("translation", [0, 0, 0]), dtype=np.float32)

    # Coerce to the shapes GNM expects, tolerating short/empty inputs.
    identity = _fit(identity, IDENTITY_DIM)
    expression = _fit(expression, EXPRESSION_DIM)
    rotations = _fit(rotations, NUM_JOINTS * 3).reshape(NUM_JOINTS, 3)
    translation = _fit(translation, 3)

    vertices = np.asarray(GNM(identity, expression, rotations, translation), dtype=np.float32)
    vertices = vertices.reshape(-1, 3)

    return jsonify({"verticesB64": b64_f32(vertices.reshape(-1))})


def _fit(arr, n):
    """Pad with zeros or truncate a 1-D array to length n."""
    arr = arr.reshape(-1)
    if arr.shape[0] == n:
        return arr
    out = np.zeros(n, dtype=np.float32)
    m = min(n, arr.shape[0])
    out[:m] = arr[:m]
    return out


# --------------------------------------------------------------------------
# Semantic samplers (lazy-loaded: TF models load on first use)
# --------------------------------------------------------------------------

import threading

_SEM = {"identity": None, "expression": None, "error": None, "backend": None}
_SEM_LOCK = threading.Lock()


def _get_samplers():
    """Load the samplers once, thread-safe. Prefers the lightweight NumPy
    backend (h5py inference, no TensorFlow); falls back to the original
    TensorFlow implementation if that fails and TF is installed."""
    with _SEM_LOCK:
        if _SEM["error"]:
            raise RuntimeError(_SEM["error"])
        if _SEM["identity"] is None:
            errors = []
            try:
                import numpy_sampler as ns
                data_dir = os.path.join(_SHAPE_DIR, "data", "semantic_sampler")
                print("Loading semantic samplers (NumPy backend) ...")
                ident, expr = ns.load(data_dir)
                _SEM.update(module=ns, identity=ident, expression=expr,
                            backend="numpy")
                print("Semantic samplers ready (NumPy — no TensorFlow).")
            except Exception as e:
                errors.append(f"numpy backend: {e}")
                try:
                    from gnm.shape import semantic_sampler as ss
                    print("Loading semantic samplers (TensorFlow backend) ...")
                    _SEM.update(module=ss, identity=ss.IdentitySampler(),
                                expression=ss.ExpressionSampler(),
                                backend="tensorflow")
                    print("Semantic samplers ready (TensorFlow).")
                except Exception as e2:
                    errors.append(f"tensorflow backend: {e2}")
                    _SEM["error"] = "semantic samplers unavailable — " +                         " | ".join(errors)
                    raise RuntimeError(_SEM["error"])
        return _SEM["module"], _SEM["identity"], _SEM["expression"]


@app.route("/api/semantic/meta")
def api_semantic_meta():
    """Enum labels for the UI. Imports the module but not the TF models."""
    try:
        try:
            import numpy_sampler as ss
        except Exception:
            from gnm.shape import semantic_sampler as ss
        return jsonify({
            "available": True,
            "backend": _SEM.get("backend") or "not loaded yet",
            "genders": [m.name for m in ss.Gender],
            "ethnicities": [m.name for m in ss.Ethnicity],
            "expressions": [m.name for m in ss.Expression],
        })
    except Exception as e:
        return jsonify({"available": False, "reason": str(e)})


@app.route("/api/semantic/identity", methods=["POST"])
def api_semantic_identity():
    """Blend-sample an identity vector from gender/ethnicity weights + seed."""
    data = request.get_json(force=True)
    try:
        ss, ident, _ = _get_samplers()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    gw_in = data.get("genderWeights", {}) or {}
    ew_in = data.get("ethnicityWeights", {}) or {}
    gender_weights = {ss.Gender[k]: float(v) for k, v in gw_in.items()
                      if k in ss.Gender.__members__ and float(v) > 0}
    ethnicity_weights = {ss.Ethnicity[k]: float(v) for k, v in ew_in.items()
                         if k in ss.Ethnicity.__members__ and float(v) > 0}
    if not gender_weights:
        gender_weights = {g: 1.0 for g in ss.Gender}
    if not ethnicity_weights:
        ethnicity_weights = {e: 1.0 for e in ss.Ethnicity}

    seed = data.get("seed")
    rng = np.random.default_rng(int(seed)) if seed is not None else None

    vec = ident.blend_identities(gender_weights, ethnicity_weights,
                                 num_samples=1, rng=rng)
    return jsonify({"identity": np.asarray(vec, dtype=np.float32).reshape(-1).tolist()})


@app.route("/api/semantic/expression", methods=["POST"])
def api_semantic_expression():
    """Sample an expression vector for a class label + seed."""
    data = request.get_json(force=True)
    try:
        ss, _, expr = _get_samplers()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    label = str(data.get("label", "HAPPY"))
    if label not in ss.Expression.__members__:
        return jsonify({"error": f"unknown expression label: {label}"}), 400

    seed = data.get("seed")
    rng = np.random.default_rng(int(seed)) if seed is not None else None

    vec = expr.sample_expression(ss.Expression[label], num_samples=1, rng=rng)
    return jsonify({"expression": np.asarray(vec, dtype=np.float32).reshape(-1).tolist()})


@app.route("/api/texture")
def api_texture():
    if not os.path.exists(TEXTURE_PATH):
        return ("texture not found", 404)
    return send_file(TEXTURE_PATH)


if __name__ == "__main__":
    print(f"Serving on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
