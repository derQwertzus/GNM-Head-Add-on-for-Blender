# GNM Head — Blender extension.
#
# A zero-dependency client for the local GNM web server (server.py from the
# gnm_web wrapper). All model evaluation and semantic sampling happens in the
# server process; this addon only ships bpy + stdlib urllib + Blender's
# bundled NumPy. Nothing is ever pip-installed into Blender.

import base64
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

import bpy
import numpy as np

# ---------------------------------------------------------------------------
# Module-level caches (rebuilt on demand; not persisted in the .blend)
# ---------------------------------------------------------------------------

META = None          # /api/meta payload (decoded)
SEMANTIC = None      # /api/semantic/meta payload
EXPR_RAW = None      # last sampled expression vector, pre-intensity
REGION_DELTA = None  # per-vertex offset from regional randomization (full-index Nx3)
VERT_SELECT = None   # full-mesh indices kept in the current (compacted) mesh, or None = all
CURRENT_FULL_VERTS = None  # last evaluated full-index vertex positions (with delta)
_ENUM_ITEMS = []     # keep EnumProperty item strings alive (Blender requirement)

_SUPPRESS = False    # guard: programmatic writes must not re-trigger updates
_DIRTY_MESH = 0.0    # timestamp of last change that needs a mesh re-eval
_DIRTY_IDENTITY = 0.0
_DIRTY_EXPR = 0.0
_DEBOUNCE = 0.3      # seconds of quiet before firing


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _server_url():
    prefs = bpy.context.preferences.addons[__package__].preferences
    return prefs.server_url.rstrip("/")


def _get(path, timeout=10):
    req = urllib.request.Request(_server_url() + path)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def _post(path, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _server_url() + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def _b64_f32(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)


def _b64_i32(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32)


# ---------------------------------------------------------------------------
# Property update callbacks (all funnel into the debounce timer)
# ---------------------------------------------------------------------------

def _mark_mesh(_self=None, _ctx=None):
    global _DIRTY_MESH
    if not _SUPPRESS:
        _DIRTY_MESH = time.monotonic()


def _mark_identity(_self=None, _ctx=None):
    global _DIRTY_IDENTITY
    if not _SUPPRESS:
        _DIRTY_IDENTITY = time.monotonic()


def _mark_expression(_self=None, _ctx=None):
    global _DIRTY_EXPR
    if not _SUPPRESS:
        _DIRTY_EXPR = time.monotonic()


def _apply_intensity(_self=None, _ctx=None):
    """Rescale the cached raw expression locally — no server round-trip."""
    if _SUPPRESS or EXPR_RAW is None:
        return
    st = bpy.context.scene.gnm_head
    _write_collection(st.expression_values, np.asarray(EXPR_RAW) * st.intensity)
    _mark_mesh()


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

class GNMFloatItem(bpy.types.PropertyGroup):
    value: bpy.props.FloatProperty(
        name="", default=0.0, soft_min=-3.0, soft_max=3.0, update=_mark_mesh,
    )


class GNMGroupItem(bpy.types.PropertyGroup):
    # A collapsible slider group: [start:end) into one of the value arrays.
    label: bpy.props.StringProperty()
    target: bpy.props.StringProperty()   # "identity" | "expression" | "rotation"
    start: bpy.props.IntProperty()
    end: bpy.props.IntProperty()
    initial: bpy.props.IntProperty(default=9)
    expanded: bpy.props.BoolProperty(default=False)
    show_all: bpy.props.BoolProperty(default=False, name="Show all")


def _expression_enum_items(_self, _ctx):
    global _ENUM_ITEMS
    if SEMANTIC and SEMANTIC.get("available"):
        _ENUM_ITEMS = [
            (n, n.lower().replace("_", " "), "") for n in SEMANTIC["expressions"]
        ]
    if not _ENUM_ITEMS:
        _ENUM_ITEMS = [("HAPPY", "happy", "")]
    return _ENUM_ITEMS


class GNMHeadState(bpy.types.PropertyGroup):
    head_object: bpy.props.PointerProperty(type=bpy.types.Object)

    # Semantic generator
    seed: bpy.props.IntProperty(
        name="Head Seed", default=0, min=0, update=_mark_identity,
    )
    seed_expression: bpy.props.IntProperty(
        name="Expression Seed", default=0, min=0, update=_mark_expression,
    )
    gender: bpy.props.FloatProperty(
        name="Female / Male", default=0.5, min=0.0, max=1.0, update=_mark_identity,
    )
    eth_middle_eastern: bpy.props.FloatProperty(
        name="Middle Eastern", default=0.25, min=0.0, max=1.0, update=_mark_identity)
    eth_asian: bpy.props.FloatProperty(
        name="Asian", default=0.25, min=0.0, max=1.0, update=_mark_identity)
    eth_white: bpy.props.FloatProperty(
        name="White", default=0.25, min=0.0, max=1.0, update=_mark_identity)
    eth_black: bpy.props.FloatProperty(
        name="Black", default=0.25, min=0.0, max=1.0, update=_mark_identity)

    expression_label: bpy.props.EnumProperty(
        name="Expression", items=_expression_enum_items, update=_mark_expression,
    )
    intensity: bpy.props.FloatProperty(
        name="Intensity", default=1.0, min=0.0, max=1.5, update=_apply_intensity,
    )

    # Full parameter arrays
    # Component visibility
    include_eyes: bpy.props.BoolProperty(
        name="Eyes", default=True, update=lambda s, c: _rebuild_mesh_safe())
    include_eye_covers: bpy.props.BoolProperty(
        name="Eye Covers", default=False,
        description="Transparent cornea shell geometry over the eyes — "
                    "occludes sclera/iris in solid shading, so off by default",
        update=lambda s, c: _rebuild_mesh_safe())
    include_teeth: bpy.props.BoolProperty(
        name="Teeth", default=True, update=lambda s, c: _rebuild_mesh_safe())
    include_tongue: bpy.props.BoolProperty(
        name="Tongue", default=True, update=lambda s, c: _rebuild_mesh_safe())

    mask_viz: bpy.props.BoolProperty(
        name="Mask Visualization", default=True,
        description="Spotlight active masks: outside turns pure black with "
                    "full roughness",
        update=lambda s, c: _update_mask_viz())

    # Regional randomization masks (vertex group names on the head object).
    # While any is active, identity sampling only affects the masked areas.
    region_eyes: bpy.props.BoolProperty(
        name="Eyes", default=False, update=lambda s, c: _update_mask_viz())
    region_nose: bpy.props.BoolProperty(
        name="Nose", default=False, update=lambda s, c: _update_mask_viz())
    region_mouth: bpy.props.BoolProperty(
        name="Mouth", default=False, update=lambda s, c: _update_mask_viz())
    region_jaw: bpy.props.BoolProperty(
        name="Jaw", default=False, update=lambda s, c: _update_mask_viz())
    region_ears: bpy.props.BoolProperty(
        name="Ears", default=False, update=lambda s, c: _update_mask_viz())
    region_back: bpy.props.BoolProperty(
        name="BackHead", default=False, update=lambda s, c: _update_mask_viz())

    identity_values: bpy.props.CollectionProperty(type=GNMFloatItem)
    expression_values: bpy.props.CollectionProperty(type=GNMFloatItem)
    rotation_values: bpy.props.CollectionProperty(type=GNMFloatItem)

    groups: bpy.props.CollectionProperty(type=GNMGroupItem)


# ---------------------------------------------------------------------------
# Collection <-> numpy helpers
# ---------------------------------------------------------------------------

def _ensure_collection(coll, n):
    while len(coll) < n:
        coll.add()
    while len(coll) > n:
        coll.remove(len(coll) - 1)


def _write_collection(coll, values):
    global _SUPPRESS
    _SUPPRESS = True
    try:
        for item, v in zip(coll, values):
            item.value = float(v)
    finally:
        _SUPPRESS = False


def _read_collection(coll):
    return [item.value for item in coll]


# ---------------------------------------------------------------------------
# Server interactions
# ---------------------------------------------------------------------------

def _fetch_meta():
    global META, SEMANTIC
    META = _get("/api/meta")
    try:
        SEMANTIC = _get("/api/semantic/meta")
    except Exception:
        SEMANTIC = {"available": False, "reason": "request failed"}
    return META


def _init_state(st):
    """Size the value collections and build the group list from META."""
    _ensure_collection(st.identity_values, META["identityDim"])
    _ensure_collection(st.expression_values, META["expressionDim"])
    _ensure_collection(st.rotation_values, META["numJoints"] * 3)

    per_group_initial = {
        "head": 9, "left eye": 3, "right eye": 3,
        "lower face": 7, "tongue": 3, "iris": 1,
    }
    st.groups.clear()
    for g in META["identityGroups"]:
        item = st.groups.add()
        item.label = g["label"]; item.target = "identity"
        item.start = g["start"]; item.end = g["end"]
        item.initial = per_group_initial.get(g["label"], 9)
    for g in META["expressionGroups"]:
        item = st.groups.add()
        item.label = g["label"]; item.target = "expression"
        item.start = g["start"]; item.end = g["end"]
        item.initial = per_group_initial.get(g["label"], 9)
    for j, name in enumerate(META["jointNames"]):
        item = st.groups.add()
        item.label = name; item.target = "rotation"
        item.start = j * 3; item.end = j * 3 + 3
        item.initial = 3


def _params_payload(st):
    return {
        "identity": _read_collection(st.identity_values),
        "expression": _read_collection(st.expression_values),
        "rotations": _read_collection(st.rotation_values),
        "translation": [0.0, 0.0, 0.0],
    }


def _eval_mesh(st):
    json_res = _post("/api/mesh", _params_payload(st))
    return _b64_f32(json_res["verticesB64"]).reshape(-1, 3)


def _post_identity(st, seed):
    return _post("/api/semantic/identity", {
        "genderWeights": {"FEMALE": 1.0 - st.gender, "MALE": st.gender},
        "ethnicityWeights": {
            "MIDDLE_EASTERN": st.eth_middle_eastern,
            "ASIAN": st.eth_asian,
            "WHITE": st.eth_white,
            "BLACK": st.eth_black,
        },
        "seed": seed,
    })


REGION_GROUP_NAMES = ("Eyes", "Nose", "Mouth", "Jaw", "Ears", "BackHead")


def _active_mask(st):
    """Combined painted mask of the checked regions, or None if none apply."""
    obj = st.head_object
    if obj is None or obj.type != "MESH":
        return None
    checked = {
        "Eyes": st.region_eyes, "Nose": st.region_nose,
        "Mouth": st.region_mouth, "Jaw": st.region_jaw,
        "Ears": st.region_ears, "BackHead": st.region_back,
    }
    names = [n for n, on in checked.items() if on]
    if not names:
        return None
    weights = _capture_vgroup_weights(obj)
    mask = np.zeros(META["numVertices"], dtype=np.float32)
    for name in names:
        w = weights.get(name)
        if w is not None:
            mask = np.maximum(mask, _expand_to_full(w))
    return mask if np.any(mask > 0) else None


def _sample_identity(st):
    """Full resample — or, when region masks are active, a masked vertex-space
    blend that leaves the base identity latents untouched."""
    global REGION_DELTA
    mask = _active_mask(st)
    if mask is None:
        res = _post_identity(st, st.seed)
        _write_collection(st.identity_values, res["identity"])
        return

    base = _eval_mesh(st)
    res = _post_identity(st, st.seed)
    payload = _params_payload(st)
    payload["identity"] = res["identity"]
    alt = _b64_f32(_post("/api/mesh", payload)["verticesB64"]).reshape(-1, 3)

    if REGION_DELTA is None or len(REGION_DELTA) != len(base):
        REGION_DELTA = np.zeros_like(base)
    REGION_DELTA = REGION_DELTA * (1.0 - mask[:, None]) + mask[:, None] * (alt - base)


def _sample_expression(st):
    global EXPR_RAW
    res = _post("/api/semantic/expression", {
        "label": st.expression_label,
        "seed": st.seed_expression,
    })
    EXPR_RAW = res["expression"]
    _write_collection(
        st.expression_values, np.asarray(EXPR_RAW) * st.intensity)


# ---------------------------------------------------------------------------
# Mesh building / updating
# ---------------------------------------------------------------------------

def _topology():
    """(faces, per-corner uvs, per-face components, corners_per_face).
    Prefers the model's native quads so Blender gets clean edge loops."""
    if "quadsB64" in META:
        faces = _b64_i32(META["quadsB64"]).reshape(-1, 4)
        uvs = _b64_f32(META["quadUvsB64"]).reshape(-1, 4, 2)
        comp = _b64_i32(META["quadComponentsB64"])
        return faces, uvs, comp, 4
    faces = _b64_i32(META["facesB64"]).reshape(-1, 3)
    uvs = _b64_f32(META["triangleUvsB64"]).reshape(-1, 3, 2)
    comp = _b64_i32(META.get("faceComponentsB64", ""))         if "faceComponentsB64" in META else None
    return faces, uvs, comp, 3


def _face_mask(st, comp, n_faces):
    """Boolean mask over faces honoring the component toggles."""
    if comp is None or len(comp) != n_faces:
        return np.ones(n_faces, dtype=bool)
    mask = np.ones(n_faces, dtype=bool)
    if not st.include_teeth:
        mask &= comp != 1
    if not st.include_tongue:
        mask &= comp != 2
    if not st.include_eyes:
        mask &= comp != 3
    if not st.include_eye_covers:
        mask &= comp != 4
    return mask


def _expand_to_full(values, fill=0.0):
    """Reduced (compacted-mesh) per-vertex array -> full-index array."""
    if VERT_SELECT is None:
        return np.asarray(values)
    full = np.full(META["numVertices"], fill, dtype=np.float32)
    full[VERT_SELECT] = values
    return full


def _reduce_from_full(values):
    """Full-index per-vertex array -> current compacted mesh order."""
    if VERT_SELECT is None:
        return np.asarray(values)
    return np.asarray(values)[VERT_SELECT]


def _make_mesh_data(st, vertices):
    """Fresh mesh datablock: filtered faces + only their vertices (no loose
    verts), per-loop UVs, vertex colors, mask attribute. Sets VERT_SELECT."""
    global VERT_SELECT
    faces, face_uvs, comp, _cpf = _topology()
    colors = _b64_f32(META["vertexColorsB64"]).reshape(-1, 3)

    fmask = _face_mask(st, comp, faces.shape[0])
    faces_f = faces[fmask]
    uvs_f = face_uvs[fmask].reshape(-1, 2)

    used = np.unique(faces_f)
    if len(used) == len(vertices):
        VERT_SELECT = None
        verts_use = np.asarray(vertices)
        faces_use = faces_f
        colors_use = colors
    else:
        VERT_SELECT = used
        verts_use = np.asarray(vertices)[used]
        faces_use = np.searchsorted(used, faces_f)
        colors_use = colors[used]

    mesh = bpy.data.meshes.new("GNM Head")
    mesh.from_pydata(verts_use.tolist(), [], faces_use.tolist())
    mesh.update()

    # Per-loop UVs: welded vertices, seams live only in UV space — natively.
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", uvs_f.reshape(-1).tolist())

    # Demo-style vertex colors as a point-domain color attribute.
    attr = mesh.color_attributes.new(
        name="GNM Color", type="FLOAT_COLOR", domain="POINT")
    rgba = np.concatenate(
        [colors_use, np.ones((len(colors_use), 1), dtype=np.float32)], axis=1)
    attr.data.foreach_set("color", rgba.reshape(-1).tolist())

    for poly in mesh.polygons:
        poly.use_smooth = True
    return mesh


def _build_head_object(st, vertices):
    global CURRENT_FULL_VERTS
    CURRENT_FULL_VERTS = np.asarray(vertices, dtype=np.float32)
    mesh = _make_mesh_data(st, vertices)
    obj = bpy.data.objects.new("GNM Head", mesh)
    # GNM is Y-up; rotate +90 deg on X so the head stands upright in Z-up Blender.
    obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    bpy.context.collection.objects.link(obj)
    st.head_object = obj
    return obj


def _capture_vgroup_weights(obj):
    """{group_name: weights ndarray} for restoring after a data swap."""
    out = {}
    mesh = obj.data
    n = len(mesh.vertices)
    for vg in obj.vertex_groups:
        w = np.zeros(n, dtype=np.float32)
        gi = vg.index
        for v in mesh.vertices:
            for g in v.groups:
                if g.group == gi:
                    w[v.index] = g.weight
                    break
        out[vg.name] = w
    return out


def _restore_vgroup_weights(obj, stash):
    for name, w in stash.items():
        vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        for i in np.nonzero(w > 0)[0]:
            vg.add([int(i)], float(w[i]), "REPLACE")


def _rebuild_mesh_safe():
    """Rebuild the head mesh with current component toggles (update callback)."""
    if _SUPPRESS or META is None:
        return
    st = bpy.context.scene.gnm_head
    obj = st.head_object
    if obj is None or obj.type != "MESH":
        return
    if CURRENT_FULL_VERTS is None:
        return
    try:
        # Capture painted weights in FULL-index space before the selection
        # changes, restore them mapped into the new selection after.
        stash_full = {name: _expand_to_full(w)
                      for name, w in _capture_vgroup_weights(obj).items()}
        old = obj.data
        obj.data = _make_mesh_data(st, CURRENT_FULL_VERTS)  # updates VERT_SELECT
        bpy.data.meshes.remove(old)
        stash_new = {name: _reduce_from_full(w) for name, w in stash_full.items()}
        _restore_vgroup_weights(obj, stash_new)
    except Exception as e:
        print(f"[GNM Head] rebuild failed: {e}")
    _update_mask_viz()


def _update_mask_viz():
    """Darken masked vertices in the color attribute while masks are active."""
    if _SUPPRESS or META is None:
        return
    st = bpy.context.scene.gnm_head
    obj = st.head_object
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    attr = mesh.color_attributes.get("GNM Color")
    if attr is None:
        return
    base = _reduce_from_full(_b64_f32(META["vertexColorsB64"]).reshape(-1, 3))
    if len(base) != len(mesh.vertices):
        return
    mask_full = _active_mask(st) if st.mask_viz else None
    colors = base.copy()
    if mask_full is not None:
        mask_vals = _reduce_from_full(mask_full)
        # Spotlight: full brightness inside the mask, dimmed to 18% outside.
        colors *= (0.18 + 0.82 * mask_vals)[:, None]
    rgba = np.concatenate(
        [colors, np.ones((len(colors), 1), dtype=np.float32)], axis=1)
    attr.data.foreach_set("color", rgba.reshape(-1).tolist())
    mesh.update()
    # Solid-mode specular highlights wash the dimmed areas out — switch them
    # off while the visualization is active, back on when it is not.
    _set_solid_specular(enabled=mask_full is None)


def _set_solid_specular(enabled):
    """Toggle Solid-shading 'Specular Lighting' in all 3D viewports; also
    ensures Solid mode while the mask visualization is active."""
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for space in area.spaces:
                    if space.type != "VIEW_3D":
                        continue
                    if not enabled:
                        space.shading.type = "SOLID"
                    space.shading.show_specular_highlight = enabled
    except Exception:
        pass


def _masks_path():
    root = bpy.utils.extension_path_user(__package__, create=True)
    return os.path.join(root, "region_masks.json")


def _default_masks_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "default_region_masks.json")


def _save_region_masks(obj):
    """Persist painted region weights in FULL-index space (canonical) —
    the topology never changes, so these apply to every generated head."""
    weights = _capture_vgroup_weights(obj)
    data = {}
    for name in REGION_GROUP_NAMES:
        w = weights.get(name)
        if w is None:
            continue
        full = _expand_to_full(w)
        nz = np.nonzero(full > 0)[0]
        data[name] = {"i": nz.tolist(), "w": full[nz].astype(float).tolist()}
    with open(_masks_path(), "w") as f:
        json.dump(data, f)
    return sum(len(d["i"]) for d in data.values())


def _load_region_masks(obj):
    """User-saved masks win; otherwise the bundled defaults ship with the
    add-on. Indices are full-index and get mapped into the current mesh."""
    data = None
    for path in (_masks_path(), _default_masks_path()):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                break
            except Exception:
                continue
    if not data:
        return 0

    n_full = META["numVertices"]
    if VERT_SELECT is None:
        inv = None
    else:
        inv = np.full(n_full, -1, dtype=np.int64)
        inv[VERT_SELECT] = np.arange(len(VERT_SELECT))

    n_mesh = len(obj.data.vertices)
    loaded = 0
    for name, d in data.items():
        vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        for i, w in zip(d.get("i", []), d.get("w", [])):
            i = int(i)
            if not 0 <= i < n_full:
                continue
            j = i if inv is None else int(inv[i])
            if 0 <= j < n_mesh:
                vg.add([j], float(w), "REPLACE")
                loaded += 1
    return loaded




def _set_viewport_vertex_colors():
    """Flip Solid-mode viewports to Attribute coloring so the colors show."""
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.color_type = "VERTEX"
    except Exception:
        pass


def _update_head_positions(st, vertices):
    """vertices: FULL-index array from the server. Applies the (full-index)
    region delta, then maps to the current compacted mesh."""
    global CURRENT_FULL_VERTS
    obj = st.head_object
    if obj is None or obj.type != "MESH":
        return False
    vertices = np.asarray(vertices, dtype=np.float32)
    if REGION_DELTA is not None and len(REGION_DELTA) == len(vertices):
        vertices = vertices + REGION_DELTA
    CURRENT_FULL_VERTS = vertices
    reduced = vertices if VERT_SELECT is None else vertices[VERT_SELECT]
    mesh = obj.data
    if len(mesh.vertices) != len(reduced):
        return False
    mesh.vertices.foreach_set("co", reduced.reshape(-1))
    mesh.update()
    return True


# ---------------------------------------------------------------------------
# Debounce timer
# ---------------------------------------------------------------------------

def _timer():
    global _DIRTY_MESH, _DIRTY_IDENTITY, _DIRTY_EXPR
    now = time.monotonic()
    try:
        scene = bpy.context.scene
        if scene is None:
            return 0.2
        st = scene.gnm_head
        if META is None or st.head_object is None:
            return 0.2
        obj = st.head_object
        active = bpy.context.view_layer.objects.active if bpy.context.view_layer else None
        if active != obj or not obj.visible_get():
            return 0.2  # UI is frozen — hold pending updates until re-selected

        did_semantic = False
        if _DIRTY_IDENTITY and now - _DIRTY_IDENTITY > _DEBOUNCE:
            _DIRTY_IDENTITY = 0.0
            if SEMANTIC and SEMANTIC.get("available"):
                _sample_identity(st)
                did_semantic = True
        if _DIRTY_EXPR and now - _DIRTY_EXPR > _DEBOUNCE:
            _DIRTY_EXPR = 0.0
            if SEMANTIC and SEMANTIC.get("available"):
                _sample_expression(st)
                did_semantic = True
        if did_semantic:
            _DIRTY_MESH = now - _DEBOUNCE - 1  # fire immediately below

        if _DIRTY_MESH and now - _DIRTY_MESH > _DEBOUNCE:
            _DIRTY_MESH = 0.0
            _update_head_positions(st, _eval_mesh(st))
    except Exception as e:
        # Server briefly unreachable etc. — try again on the next tick.
        print(f"[GNM Head] update skipped: {e}")
    return 0.15


# ---------------------------------------------------------------------------
# Managed environment + server process
# ---------------------------------------------------------------------------

GNM_TARBALL_URL = "https://github.com/google/GNM/archive/refs/heads/main.tar.gz"

SERVER_PROC = None
INSTALL_THREAD = None
INSTALL_STATE = ""       # short status line shown in preferences
INSTALL_ERROR = ""


def _managed_root():
    return bpy.utils.extension_path_user(__package__, create=True)


def _env_dir():
    return os.path.join(_managed_root(), "env")


def _src_dir():
    return os.path.join(_managed_root(), "GNM")


def _env_python():
    if platform.system() == "Windows":
        return os.path.join(_env_dir(), "Scripts", "python.exe")
    return os.path.join(_env_dir(), "bin", "python")


def _install_marker():
    return os.path.join(_managed_root(), ".installed")


def _env_installed():
    return os.path.exists(_env_python()) and os.path.exists(_install_marker())


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _fmt_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024


def _bundled_server_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")


# Minimal runtime set: the server + gnm_numpy + h5py-based semantic sampling.
# TensorFlow is NOT required — the bundled numpy_sampler runs the .h5 decoders.
CORE_DEPS = ["numpy", "h5py", "flask", "absl-py", "immutabledict",
             "typeguard", "opt-einsum", "etils[epath]"]
PRUNE_DIRS = ("demos", "notebooks", ".github")


def _prune_sources():
    """Drop tests/demos/notebooks from the extracted tree — dead weight for a
    headless server."""
    removed = 0
    for root, dirs, files in os.walk(_src_dir(), topdown=True):
        for d in list(dirs):
            if d in PRUNE_DIRS:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
                removed += 1
        for f in files:
            if f.endswith("_test.py") or f.endswith(".ipynb"):
                try:
                    os.remove(os.path.join(root, f))
                    removed += 1
                except OSError:
                    pass
    return removed


def _install_worker(base_python, with_tf):
    """Runs in a thread: venv + GNM sources + minimal pip installs, logged."""
    global INSTALL_STATE, INSTALL_ERROR
    INSTALL_ERROR = ""
    root = _managed_root()
    log_path = os.path.join(root, "install.log")

    def run(step, cmd, check=True, **kw):
        global INSTALL_STATE
        INSTALL_STATE = step
        with open(log_path, "a") as log:
            log.write(f"\n==== {step}: {' '.join(cmd)}\n")
            log.flush()
            res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, **kw)
        if check and res.returncode != 0:
            raise RuntimeError(f"{step} failed (see install.log)")
        return res.returncode

    try:
        open(log_path, "w").write("GNM Head environment install\n")

        INSTALL_STATE = "creating virtual environment"
        run("venv", [base_python, "-m", "venv", _env_dir()])

        INSTALL_STATE = "downloading GNM sources (github.com/google/GNM)"
        tar_path = os.path.join(root, "gnm_src.tar.gz")
        urllib.request.urlretrieve(GNM_TARBALL_URL, tar_path)

        INSTALL_STATE = "extracting sources"
        if os.path.exists(_src_dir()):
            shutil.rmtree(_src_dir())
        with tarfile.open(tar_path) as tar:
            tmp = os.path.join(root, "_extract")
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            tar.extractall(tmp)
            inner = next(os.scandir(tmp)).path  # GNM-main/
            shutil.move(inner, _src_dir())
            shutil.rmtree(tmp, ignore_errors=True)
        os.remove(tar_path)

        INSTALL_STATE = "pruning tests/demos"
        _prune_sources()

        py = _env_python()
        pipi = [py, "-m", "pip", "install", "--no-cache-dir"]
        run("upgrading pip", pipi + ["--upgrade", "pip"])

        deps = list(CORE_DEPS) + (["tensorflow"] if with_tf else [])
        INSTALL_STATE = "installing core dependencies (PyPI)" +             (" + tensorflow" if with_tf else " — no TensorFlow")
        run("pip core deps", pipi + deps)

        INSTALL_STATE = "installing gnm-shape (no extra deps)"
        run("pip gnm-shape", pipi + ["--no-deps", "-e",
                                     os.path.join(_src_dir(), "gnm", "shape")])

        # Auto-heal: if gnm_numpy needs a module we did not predict, install
        # exactly that and retry — keeps the footprint minimal by construction.
        INSTALL_STATE = "verifying imports"
        for _attempt in range(6):
            probe = subprocess.run(
                [py, "-c", "import gnm.shape.gnm_numpy"],
                capture_output=True, text=True)
            if probe.returncode == 0:
                break
            missing = None
            for line in probe.stderr.splitlines():
                if "No module named" in line:
                    missing = line.split("'")[1].split(".")[0]
            if not missing:
                with open(log_path, "a") as log:
                    log.write("\n==== import probe stderr:\n" + probe.stderr)
                raise RuntimeError("gnm import failed (see install.log)")
            run(f"installing missing dependency: {missing}", pipi + [missing])
        else:
            raise RuntimeError("gnm import still failing after auto-heal")

        open(_install_marker(), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        INSTALL_STATE = "done"
    except Exception as e:
        INSTALL_ERROR = str(e)
        INSTALL_STATE = f"error: {e}"


def _server_running():
    return SERVER_PROC is not None and SERVER_PROC.poll() is None


def _start_server(console=False):
    """Launch the bundled server with the managed environment's python.

    console=True opens a visible cmd window with the live log (Windows);
    otherwise the process is hidden and logs to server.log (reset each start).
    """
    global SERVER_PROC
    if _server_running():
        return True
    if not _env_installed():
        return False
    cmd = [_env_python(), "-u", os.path.join(_bundled_server_dir(), "server.py")]
    if console and platform.system() == "Windows":
        SERVER_PROC = subprocess.Popen(
            cmd, cwd=_bundled_server_dir(),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    else:
        log = open(os.path.join(_managed_root(), "server.log"), "w")
        flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        SERVER_PROC = subprocess.Popen(
            cmd, cwd=_bundled_server_dir(),
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    return True


def _stop_server():
    global SERVER_PROC
    if SERVER_PROC is not None:
        try:
            SERVER_PROC.terminate()
            SERVER_PROC.wait(timeout=5)
        except Exception:
            try:
                SERVER_PROC.kill()
            except Exception:
                pass
        SERVER_PROC = None


def _wait_for_server(timeout=90):
    """Poll /api/meta until the model has loaded (first start takes a while)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            _get("/api/meta", timeout=3)
            return True
        except Exception:
            if not _server_running():
                return False
            time.sleep(1.5)
    return False


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class GNMHEAD_OT_generate(bpy.types.Operator):
    """Connect to the GNM server and create (or refresh) the head object"""
    bl_idname = "gnmhead.generate"
    bl_label = "Generate Head"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        st = context.scene.gnm_head
        try:
            _fetch_meta()
        except Exception:
            prefs = context.preferences.addons[__package__].preferences
            if prefs.auto_start and _env_installed():
                self.report({"INFO"}, "Starting managed GNM server ...")
                _start_server(console=prefs.show_server_console)
                if not _wait_for_server():
                    self.report({"ERROR"},
                                "Managed server did not come up — see server.log "
                                "in the add-on's user directory")
                    return {"CANCELLED"}
                try:
                    _fetch_meta()
                except Exception as e:
                    self.report({"ERROR"}, f"Cannot reach GNM server: {e}")
                    return {"CANCELLED"}
            else:
                self.report({"ERROR"},
                            "Cannot reach GNM server — start it, or install the "
                            "managed environment in the add-on preferences")
                return {"CANCELLED"}

        _init_state(st)
        try:
            vertices = _eval_mesh(st)
        except Exception as e:
            self.report({"ERROR"}, f"Mesh evaluation failed: {e}")
            return {"CANCELLED"}

        if st.head_object is None or st.head_object.name not in context.scene.objects:
            _build_head_object(st, vertices)
        elif not _update_head_positions(st, vertices):
            _build_head_object(st, vertices)

        obj = st.head_object
        _set_viewport_vertex_colors()
        loaded = _load_region_masks(obj)
        _update_mask_viz()

        msg = f"GNM head ready ({META['numVertices']} verts)"
        if loaded:
            msg += " — saved region masks applied"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class GNMHEAD_OT_neutral(bpy.types.Operator):
    """Reset the expression to neutral"""
    bl_idname = "gnmhead.neutral"
    bl_label = "Neutral"

    def execute(self, context):
        global EXPR_RAW
        st = context.scene.gnm_head
        EXPR_RAW = None
        _write_collection(st.expression_values,
                          np.zeros(len(st.expression_values)))
        _mark_mesh()
        return {"FINISHED"}


class GNMHEAD_OT_reset_all(bpy.types.Operator):
    """Reset all parameters to the template head"""
    bl_idname = "gnmhead.reset_all"
    bl_label = "Reset All"

    def execute(self, context):
        global EXPR_RAW
        st = context.scene.gnm_head
        EXPR_RAW = None
        for coll in (st.identity_values, st.expression_values, st.rotation_values):
            _write_collection(coll, np.zeros(len(coll)))
        _mark_mesh()
        return {"FINISHED"}


class GNMHEAD_OT_reseed_identity(bpy.types.Operator):
    """New random head: reseeds and resamples the base identity only"""
    bl_idname = "gnmhead.reseed_identity"
    bl_label = "New Head"

    def execute(self, context):
        context.scene.gnm_head.seed = random.randint(0, 99999)
        return {"FINISHED"}


class GNMHEAD_OT_reseed_expression(bpy.types.Operator):
    """New random expression: reseeds and resamples the expression only"""
    bl_idname = "gnmhead.reseed_expression"
    bl_label = "New Expression"

    def execute(self, context):
        context.scene.gnm_head.seed_expression = random.randint(0, 99999)
        return {"FINISHED"}


class GNMHEAD_OT_create_region_groups(bpy.types.Operator):
    """Create empty vertex groups for the region masks (weight-paint them)"""
    bl_idname = "gnmhead.create_region_groups"
    bl_label = "Create Region Groups"

    def execute(self, context):
        obj = context.scene.gnm_head.head_object
        if obj is None:
            self.report({"ERROR"}, "No head object")
            return {"CANCELLED"}
        created = 0
        for name in REGION_GROUP_NAMES:
            if name not in obj.vertex_groups:
                obj.vertex_groups.new(name=name)
                created += 1
        self.report({"INFO"},
                    f"{created} group(s) created — weight-paint them to define the regions")
        return {"FINISHED"}


class GNMHEAD_OT_save_masks(bpy.types.Operator):
    """Save the painted region masks; every new head auto-loads them"""
    bl_idname = "gnmhead.save_masks"
    bl_label = "Save Region Masks"

    def execute(self, context):
        obj = context.scene.gnm_head.head_object
        if obj is None:
            self.report({"ERROR"}, "No head object")
            return {"CANCELLED"}
        count = _save_region_masks(obj)
        if count == 0:
            self.report({"WARNING"}, "No painted weights found to save")
        else:
            self.report({"INFO"}, f"Saved masks ({count} weighted vertices)")
        return {"FINISHED"}


class GNMHEAD_OT_clear_regions(bpy.types.Operator):
    """Remove all masked identity changes, back to the pure parametric head"""
    bl_idname = "gnmhead.clear_regions"
    bl_label = "Clear Masked Changes"

    def execute(self, context):
        global REGION_DELTA
        REGION_DELTA = None
        _mark_mesh()
        return {"FINISHED"}


class GNMHEAD_OT_bake(bpy.types.Operator):
    """Copy the current head as an independent mesh object in the scene"""
    bl_idname = "gnmhead.bake"
    bl_label = "Bake Identity"

    def execute(self, context):
        st = context.scene.gnm_head
        src = st.head_object
        if src is None:
            self.report({"ERROR"}, "No head to bake")
            return {"CANCELLED"}
        dup = src.copy()
        dup.data = src.data.copy()
        dup.name = "GNM Head Identity"
        dup.data.name = "GNM Head Identity"

        # Bake clean: original vertex colors, no mask spotlight/roughness.
        base = _reduce_from_full(
            _b64_f32(META["vertexColorsB64"]).reshape(-1, 3))
        attr = dup.data.color_attributes.get("GNM Color")
        if attr is not None and len(base) == len(dup.data.vertices):
            rgba = np.concatenate(
                [base, np.ones((len(base), 1), dtype=np.float32)], axis=1)
            attr.data.foreach_set("color", rgba.reshape(-1).tolist())

        context.collection.objects.link(dup)
        for o in context.selected_objects:
            o.select_set(False)
        dup.select_set(True)
        context.view_layer.objects.active = dup
        self.report({"INFO"}, f"Baked: {dup.name}")
        return {"FINISHED"}


class GNMHEAD_OT_install_env(bpy.types.Operator):
    """Create the managed environment: venv + GNM from GitHub + deps from PyPI (~2 GB)"""
    bl_idname = "gnmhead.install_env"
    bl_label = "Install Environment"

    _timer = None

    def execute(self, context):
        global INSTALL_THREAD
        if INSTALL_THREAD is not None and INSTALL_THREAD.is_alive():
            self.report({"WARNING"}, "Install already running")
            return {"CANCELLED"}
        prefs = context.preferences.addons[__package__].preferences
        base = prefs.base_python.strip() or "python"
        if shutil.which(base) is None and not os.path.exists(base):
            self.report({"ERROR"},
                        f"Python not found: '{base}' — set Base Python in preferences "
                        "(needs Python 3.10-3.12 installed on the system)")
            return {"CANCELLED"}
        INSTALL_THREAD = threading.Thread(
            target=_install_worker, args=(base, prefs.install_tensorflow),
            daemon=True)
        INSTALL_THREAD.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "TIMER":
            for area in context.screen.areas:
                area.tag_redraw()
            if INSTALL_THREAD is None or not INSTALL_THREAD.is_alive():
                context.window_manager.event_timer_remove(self._timer)
                if INSTALL_ERROR:
                    self.report({"ERROR"}, f"Install failed: {INSTALL_ERROR}")
                else:
                    self.report({"INFO"}, "GNM environment installed")
                return {"FINISHED"}
        return {"PASS_THROUGH"}


class GNMHEAD_OT_uninstall_env(bpy.types.Operator):
    """Delete the entire managed folder: environment, GNM sources, logs and
    saved region masks"""
    bl_idname = "gnmhead.uninstall_env"
    bl_label = "Uninstall Environment"
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        _stop_server()
        root = _managed_root()
        # The server.log handle is released once the process is gone; give
        # Windows a beat, then retry once if the tree is still locked.
        for attempt in range(2):
            shutil.rmtree(root, ignore_errors=(attempt == 0))
            if not os.path.exists(root) or not os.listdir(root):
                break
            time.sleep(1.0)
        if os.path.exists(root) and os.listdir(root):
            self.report({"WARNING"},
                        f"Some files could not be removed (still locked?): {root}")
        else:
            self.report({"INFO"}, "Managed folder removed entirely")
        return {"FINISHED"}


class GNMHEAD_OT_start_server(bpy.types.Operator):
    """Start the bundled GNM server using the managed environment"""
    bl_idname = "gnmhead.start_server"
    bl_label = "Start Server"

    def execute(self, context):
        if _server_running():
            self.report({"INFO"}, "Server already running")
            return {"FINISHED"}
        if not _env_installed():
            self.report({"ERROR"}, "Install the environment first (add-on preferences)")
            return {"CANCELLED"}
        prefs = context.preferences.addons[__package__].preferences
        _start_server(console=prefs.show_server_console)
        self.report({"INFO"}, "Server starting — first start loads the model (10-30 s)")
        return {"FINISHED"}


class GNMHEAD_OT_stop_server(bpy.types.Operator):
    """Stop the bundled GNM server"""
    bl_idname = "gnmhead.stop_server"
    bl_label = "Stop Server"

    def execute(self, context):
        if not _server_running():
            self.report({"INFO"}, "Server is not running")
            return {"FINISHED"}
        _stop_server()
        self.report({"INFO"}, "Server stopped")
        return {"FINISHED"}


class GNMHEAD_OT_test_connection(bpy.types.Operator):
    """Check the GNM server and report model info"""
    bl_idname = "gnmhead.test_connection"
    bl_label = "Test Connection"

    def execute(self, context):
        try:
            meta = _get("/api/meta", timeout=5)
            sem = _get("/api/semantic/meta", timeout=5)
            sem_txt = "semantic sampling available" if sem.get("available") \
                else "semantic sampling unavailable"
            self.report({"INFO"},
                        f"OK — {meta['numVertices']} verts, "
                        f"{meta['identityDim']}+{meta['expressionDim']} params, {sem_txt}")
        except Exception as e:
            self.report({"ERROR"}, f"Server unreachable: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

class GNMHeadPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    server_url: bpy.props.StringProperty(
        name="Server URL",
        default="http://127.0.0.1:5000",
        description="Address of the GNM server (managed or external)",
    )
    base_python: bpy.props.StringProperty(
        name="Base Python",
        default="python",
        description="System Python 3.10-3.12 used to create the managed "
                    "environment (command name or full path)",
    )
    install_tensorflow: bpy.props.BoolProperty(
        name="Also install TensorFlow (fallback backend, ~2 GB)",
        default=False,
        description="Semantic sampling normally runs on the bundled NumPy "
                    "backend (~no extra weight). Enable only if the NumPy "
                    "backend reports an unsupported layer",
    )
    show_server_console: bpy.props.BoolProperty(
        name="Open server in a console window",
        default=True,
        description="Start the managed server in a visible cmd window with the "
                    "live log. When off, it runs hidden and logs to server.log",
    )
    auto_start: bpy.props.BoolProperty(
        name="Auto-start managed server",
        default=True,
        description="Start the bundled server automatically when Generate Head "
                    "cannot reach one",
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "server_url")
        row = col.row(align=True)
        row.operator("gnmhead.test_connection", icon="URL")
        if _server_running():
            row.operator("gnmhead.stop_server", icon="PAUSE")
        else:
            row.operator("gnmhead.start_server", icon="PLAY")
        col.prop(self, "auto_start")
        col.prop(self, "show_server_console")

        box = col.box()
        box.label(text="Managed Environment", icon="PACKAGE")
        installing = INSTALL_THREAD is not None and INSTALL_THREAD.is_alive()

        if installing:
            box.label(text=f"Installing: {INSTALL_STATE} ...", icon="TIME")
        elif _env_installed():
            env_size = _dir_size(_env_dir())
            src_size = _dir_size(_src_dir())
            box.label(text="Installed", icon="CHECKMARK")
            box.label(text=f"  Environment: {_fmt_size(env_size)}  —  {_env_dir()}")
            box.label(text=f"  GNM sources: {_fmt_size(src_size)}  —  {_src_dir()}")
            box.label(text=f"  Logs: install.log / server.log in {_managed_root()}")
            box.operator("gnmhead.uninstall_env", icon="TRASH")
        else:
            if INSTALL_ERROR:
                box.label(text=f"Last install failed: {INSTALL_ERROR}", icon="ERROR")
            box.label(text="Not installed. Downloads from original sources:")
            box.label(text="  GNM: github.com/google/GNM (Apache-2.0)")
            box.label(text="  NumPy, h5py, Flask etc.: PyPI (~170 MB total)")
            box.prop(self, "base_python")
            box.prop(self, "install_tensorflow")
            row = box.row()
            row.scale_y = 1.4
            row.operator("gnmhead.install_env", icon="IMPORT")

        info = col.box()
        info.label(text="Add-on itself installs nothing into Blender's Python:", icon="INFO")
        info.label(text=f"  Blender Python: {sys.executable}")
        info.label(text=f"  Bundled NumPy {np.__version__} (read-only use)")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _head_active(context):
    """UI is live only while the GNM head is the active, visible object."""
    st = context.scene.gnm_head
    obj = st.head_object
    return (obj is not None and META is not None
            and context.view_layer.objects.active == obj
            and obj.visible_get())

class GNMHEAD_PT_main(bpy.types.Panel):
    bl_label = "GNM Head"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Head"

    def draw(self, context):
        st = context.scene.gnm_head
        col = self.layout.column()
        has_head = st.head_object is not None and META is not None

        if not has_head:
            col.operator("gnmhead.generate", text="Generate Head",
                         icon="MESH_MONKEY")
            col.label(text="Server controls are in the GNM Server panel.",
                      icon="INFO")
            return

        if not _head_active(context):
            col.label(text="Select the GNM Head object to edit.", icon="RESTRICT_SELECT_ON")
            return

        col.operator("gnmhead.reset_all", icon="LOOP_BACK")

        bake = col.column()
        bake.scale_y = 1.6
        bake.operator("gnmhead.bake", icon="DUPLICATE")


class GNMHEAD_PT_server(bpy.types.Panel):
    bl_label = "GNM Server"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Head"
    bl_parent_id = "GNMHEAD_PT_main"

    def draw(self, context):
        prefs = context.preferences.addons[__package__].preferences
        col = self.layout.column()
        if _server_running():
            col.label(text="Managed server: running", icon="CHECKMARK")
        elif _env_installed():
            col.label(text="Managed server: stopped", icon="X")
        else:
            col.label(text="No managed environment (see preferences)", icon="ERROR")
        row = col.row(align=True)
        if _server_running():
            row.operator("gnmhead.stop_server", icon="PAUSE")
        else:
            row.operator("gnmhead.start_server", icon="PLAY")
        row.operator("gnmhead.test_connection", text="", icon="URL")
        col.prop(prefs, "show_server_console")
        col.operator("gnmhead.generate", text="Refresh / Reconnect",
                     icon="FILE_REFRESH")


class GNMHEAD_PT_generate(bpy.types.Panel):
    bl_label = "Generate"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Head"
    bl_parent_id = "GNMHEAD_PT_main"

    @classmethod
    def poll(cls, context):
        return _head_active(context)

    def draw(self, context):
        st = context.scene.gnm_head
        col = self.layout.column()

        if not (SEMANTIC and SEMANTIC.get("available")):
            col.label(text="Semantic samplers unavailable on server.", icon="ERROR")
            return

        col.label(text="First sample loads models (may pause briefly).", icon="INFO")

        col.separator()
        col.label(text="Identity")
        row = col.row(align=True)
        row.prop(st, "seed")
        row.operator("gnmhead.reseed_identity", text="", icon="FILE_REFRESH")
        col.prop(st, "gender", slider=True)
        col.prop(st, "eth_middle_eastern", slider=True)
        col.prop(st, "eth_asian", slider=True)
        col.prop(st, "eth_white", slider=True)
        col.prop(st, "eth_black", slider=True)

        col.separator()
        col.label(text="Expression")
        row = col.row(align=True)
        row.prop(st, "seed_expression")
        row.operator("gnmhead.reseed_expression", text="", icon="FILE_REFRESH")
        col.prop(st, "expression_label", text="")
        col.prop(st, "intensity", slider=True)
        col.operator("gnmhead.neutral", icon="X")

        col.separator()
        col.label(text="Components")
        row = col.row(align=True)
        row.prop(st, "include_eyes", toggle=True)
        row.prop(st, "include_teeth", toggle=True)
        row = col.row(align=True)
        row.prop(st, "include_tongue", toggle=True)
        row.prop(st, "include_eye_covers", toggle=True)


def _draw_group_panel(layout, st, target):
    for gi, g in enumerate(st.groups):
        if g.target != target:
            continue
        box = layout.box()
        header = box.row()
        header.prop(g, "expanded", text=g.label,
                    icon="TRIA_DOWN" if g.expanded else "TRIA_RIGHT",
                    emboss=False)
        header.label(text=str(g.end - g.start))
        if not g.expanded:
            continue

        coll = getattr(st, f"{target}_values")
        names = None
        if META:
            names = META.get(
                {"identity": "identityNames", "expression": "expressionNames"}
                .get(target))
        count = g.end - g.start
        shown = count if g.show_all else min(g.initial, count)
        sub = box.column(align=True)
        for k in range(shown):
            idx = g.start + k
            if target == "rotation":
                label = "XYZ"[k % 3]
            elif names and idx < len(names):
                label = names[idx]
            else:
                label = f"{target[0]}{idx}"
            sub.prop(coll[idx], "value", text=label)
        if count > g.initial:
            box.prop(g, "show_all", toggle=True,
                     text=("Show less" if g.show_all else f"Show all {count}"))


class GNMHEAD_PT_regions(bpy.types.Panel):
    bl_label = "Mask Regions"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Head"
    bl_parent_id = "GNMHEAD_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _head_active(context)

    def draw(self, context):
        st = context.scene.gnm_head
        col = self.layout.column()
        row = col.row(align=True)
        row.operator("gnmhead.create_region_groups", icon="GROUP_VERTEX")
        row.operator("gnmhead.save_masks", text="", icon="FILE_TICK")
        col.prop(st, "mask_viz", toggle=True, icon="HIDE_OFF")
        col.label(text="Active masks stay bright, the rest goes black;")
        col.label(text="identity randomize only affects bright areas:")
        sub = col.column(align=True)
        sub.prop(st, "region_eyes", toggle=True)
        sub.prop(st, "region_nose", toggle=True)
        sub.prop(st, "region_mouth", toggle=True)
        sub.prop(st, "region_jaw", toggle=True)
        sub.prop(st, "region_ears", toggle=True)
        sub.prop(st, "region_back", toggle=True)
        col.separator()
        col.operator("gnmhead.clear_regions", icon="X")
        col.label(text="Masked changes live in memory — bake to keep.", icon="INFO")


class GNMHEAD_PT_identity(bpy.types.Panel):
    bl_label = "Identity Advanced"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Head"
    bl_parent_id = "GNMHEAD_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _head_active(context)

    def draw(self, context):
        _draw_group_panel(self.layout, context.scene.gnm_head, "identity")


class GNMHEAD_PT_expression(bpy.types.Panel):
    bl_label = "Expression Advanced"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Head"
    bl_parent_id = "GNMHEAD_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _head_active(context)

    def draw(self, context):
        _draw_group_panel(self.layout, context.scene.gnm_head, "expression")


class GNMHEAD_PT_pose(bpy.types.Panel):
    bl_label = "Head Pose"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GNM Head"
    bl_parent_id = "GNMHEAD_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return _head_active(context)

    def draw(self, context):
        _draw_group_panel(self.layout, context.scene.gnm_head, "rotation")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_CLASSES = (
    GNMFloatItem,
    GNMGroupItem,
    GNMHeadState,
    GNMHEAD_OT_generate,
    GNMHEAD_OT_neutral,
    GNMHEAD_OT_reset_all,
    GNMHEAD_OT_reseed_identity,
    GNMHEAD_OT_reseed_expression,
    GNMHEAD_OT_create_region_groups,
    GNMHEAD_OT_save_masks,
    GNMHEAD_OT_clear_regions,
    GNMHEAD_OT_bake,
    GNMHEAD_OT_install_env,
    GNMHEAD_OT_uninstall_env,
    GNMHEAD_OT_start_server,
    GNMHEAD_OT_stop_server,
    GNMHEAD_OT_test_connection,
    GNMHeadPreferences,
    GNMHEAD_PT_main,
    GNMHEAD_PT_server,
    GNMHEAD_PT_generate,
    GNMHEAD_PT_regions,
    GNMHEAD_PT_identity,
    GNMHEAD_PT_expression,
    GNMHEAD_PT_pose,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gnm_head = bpy.props.PointerProperty(type=GNMHeadState)
    bpy.app.timers.register(_timer, first_interval=0.5, persistent=True)


def unregister():
    _stop_server()
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
    del bpy.types.Scene.gnm_head
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
