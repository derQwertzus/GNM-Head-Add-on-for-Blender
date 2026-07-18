"""NumPy inference for the GNM semantic sampler decoders.

Replaces TensorFlow for sampling: the .h5 files are small Keras CVAE decoders
(Dense stacks), and everything around the decoder call is already NumPy in the
original gnm.shape.semantic_sampler. This module loads the Keras-v2 HDF5
files with h5py and executes the layer graph directly.

Supported layers: InputLayer, Dense, Concatenate, Activation, ReLU,
LeakyReLU, BatchNormalization, Dropout (identity), Flatten (2-D identity).
Anything else raises with a clear message — in that case install the
TensorFlow fallback tier.

Enums and public methods mirror gnm.shape.semantic_sampler so the server can
use either backend interchangeably.
"""

import enum
import json
import os

import h5py
import numpy as np


# --- Enums (mirroring gnm.shape.semantic_sampler exactly) -------------------

class Gender(enum.IntEnum):
    FEMALE = 0
    MALE = 1


class Ethnicity(enum.IntEnum):
    MIDDLE_EASTERN = 0
    ASIAN = 1
    WHITE = 2
    BLACK = 3


class Expression(enum.IntEnum):
    SURPRISE = 0
    DISGUST = 1
    SUCK = 2
    COMPRESS_FACE = 3
    STRETCH_FACE = 4
    HAPPY = 5
    SQUINT = 6
    PLATYSMA = 7
    BLOW = 8
    FUNNELER = 9
    SMILE_WIDE = 10
    CORNERS_DOWN = 11
    PUCKER = 12
    WINK_LEFT = 13
    WINK_RIGHT = 14
    MOUTH_LEFT = 15
    MOUTH_RIGHT = 16
    LIPS_ROLL_IN = 17
    SNARL = 18
    TONGUE_CENTER = 19


# --- Activations -------------------------------------------------------------

def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


_ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "elu": lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0),
    "selu": lambda x: 1.05070098 * np.where(
        x > 0, x, 1.67326324 * (np.exp(np.minimum(x, 0)) - 1.0)),
    "softplus": lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0),
    "softmax": _softmax,
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(
        0.7978845608 * (x + 0.044715 * x ** 3))),
    "swish": lambda x: x / (1.0 + np.exp(-x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.2 * x),
}


def _decode(s):
    return s.decode("utf-8") if isinstance(s, bytes) else s


class KerasH5Decoder:
    """Loads a Keras-v2 .h5 functional model and runs a NumPy forward pass."""

    def __init__(self, path):
        self.path = path
        with h5py.File(path, "r") as f:
            raw = f.attrs.get("model_config")
            if raw is None:
                raise ValueError(f"{path}: no model_config attribute "
                                 "(not a Keras-v2 .h5 save)")
            cfg = json.loads(_decode(raw))
            self.config = cfg["config"]
            self.weights = self._load_weights(f)

        self.layers = {l["name"]: l for l in self.config["layers"]}
        self.order = [l["name"] for l in self.config["layers"]]
        self.input_names = [io[0] for io in self.config["input_layers"]]
        self.output_name = self.config["output_layers"][0][0]
        self.input_dims = [self._input_dim(n) for n in self.input_names]

    def _input_dim(self, name):
        conf = self.layers[name]["config"]
        shape = conf.get("batch_input_shape") or conf.get("batch_shape")
        return int(shape[-1]) if shape else None

    @staticmethod
    def _load_weights(f):
        out = {}
        if "model_weights" not in f:
            raise ValueError("no model_weights group in .h5")
        mw = f["model_weights"]
        for layer_name in mw:
            grp = mw[layer_name]
            names = [_decode(n) for n in grp.attrs.get("weight_names", [])]
            arrays = {}
            for wname in names:
                short = wname.split("/")[-1].split(":")[0]  # kernel, bias, gamma...
                arrays[short] = np.array(grp[wname], dtype=np.float32)
            if arrays:
                out[layer_name] = arrays
        return out

    def _inbound(self, layer):
        """Names of the layers feeding this one (Keras-v2 node format)."""
        nodes = layer.get("inbound_nodes", [])
        if not nodes:
            return []
        node = nodes[0]
        if isinstance(node, dict):  # Keras 3 style — not produced by these files
            raise ValueError(
                f"unsupported (Keras 3) node format in layer {layer['name']}")
        return [entry[0] for entry in node]

    def __call__(self, inputs):
        """inputs: list of arrays matching the model's input layers order."""
        values = {}
        for name, arr, dim in zip(self.input_names, inputs, self.input_dims):
            arr = np.asarray(arr, dtype=np.float32)
            if dim is not None and arr.shape[-1] != dim:
                raise ValueError(
                    f"input '{name}' expects dim {dim}, got {arr.shape[-1]}")
            values[name] = arr

        for name in self.order:
            layer = self.layers[name]
            cls = layer["class_name"]
            conf = layer["config"]
            if cls == "InputLayer":
                continue
            srcs = [values[n] for n in self._inbound(layer)]
            w = self.weights.get(name, {})

            if cls == "Dense":
                x = srcs[0] @ w["kernel"]
                if conf.get("use_bias", True) and "bias" in w:
                    x = x + w["bias"]
                act = conf.get("activation", "linear")
                if act not in _ACTIVATIONS:
                    raise ValueError(f"unsupported activation '{act}' in {name}")
                x = _ACTIVATIONS[act](x)
            elif cls == "Concatenate":
                x = np.concatenate(srcs, axis=conf.get("axis", -1))
            elif cls == "Activation":
                act = conf.get("activation", "linear")
                if act not in _ACTIVATIONS:
                    raise ValueError(f"unsupported activation '{act}' in {name}")
                x = _ACTIVATIONS[act](srcs[0])
            elif cls == "ReLU":
                x = np.maximum(srcs[0], 0.0)
                mx = conf.get("max_value")
                if mx is not None:
                    x = np.minimum(x, mx)
            elif cls == "LeakyReLU":
                alpha = conf.get("alpha", conf.get("negative_slope", 0.3))
                x = np.where(srcs[0] > 0, srcs[0], alpha * srcs[0])
            elif cls == "BatchNormalization":
                eps = conf.get("epsilon", 1e-3)
                mean = w["moving_mean"]
                var = w["moving_variance"]
                x = (srcs[0] - mean) / np.sqrt(var + eps)
                if "gamma" in w:
                    x = x * w["gamma"]
                if "beta" in w:
                    x = x + w["beta"]
            elif cls in ("Dropout", "GaussianNoise"):
                x = srcs[0]  # inference: identity
            elif cls == "Flatten":
                x = srcs[0].reshape(srcs[0].shape[0], -1)
            else:
                raise ValueError(
                    f"unsupported layer type '{cls}' ({name}) — "
                    "install the TensorFlow fallback tier")
            values[name] = x

        return values[self.output_name]

    # keras-compatible call used by the sampler code
    def predict(self, inputs, verbose=0):
        return self(inputs)


# --- Samplers (mirroring gnm.shape.semantic_sampler math) -------------------

def _get_rng(rng):
    return rng if rng is not None else np.random.default_rng()


def _one_hot(indices, num_classes):
    return np.eye(num_classes, dtype=np.float32)[np.asarray(indices, dtype=int)]


class ExpressionSampler:
    def __init__(self, decoder_model_path):
        self._decoder = KerasH5Decoder(decoder_model_path)
        self._latent_dim = self._decoder.input_dims[0]
        self._num_classes = self._decoder.input_dims[1]

    def sample_expression(self, class_label, num_samples=1, rng=None,
                          verbose=False):
        one_hot = np.repeat(_one_hot([int(class_label)], self._num_classes),
                            num_samples, axis=0)
        rng = _get_rng(rng)
        z = rng.normal(size=(num_samples, self._latent_dim)).astype(np.float32)
        return self._decoder.predict([z, one_hot])

    def blend_expressions(self, class_weights, rng=None, verbose=False):
        if not class_weights:
            raise ValueError("class_weights cannot be empty")
        total = sum(class_weights.values())
        if np.isclose(total, 0):
            raise ValueError("Sum of class_weights cannot be 0")
        norm = {int(k): v / total for k, v in class_weights.items()}

        rng = _get_rng(rng)
        z_blend = np.zeros((1, self._latent_dim), dtype=np.float32)
        oh_blend = np.zeros((1, self._num_classes), dtype=np.float32)
        for idx, w in norm.items():
            z = rng.normal(size=(1, self._latent_dim)).astype(np.float32)
            z_blend += z * w
            oh_blend += _one_hot([idx], self._num_classes) * w
        return self._decoder.predict([z_blend, oh_blend])


class IdentitySampler:
    _NUM_GENDER_CLASSES = len(Gender)
    _NUM_ETHNICITIES_CLASSES = len(Ethnicity)

    def __init__(self, decoder_model_path):
        self._decoder = KerasH5Decoder(decoder_model_path)
        self._LATENT_DIM = self._decoder.input_dims[0]

    def sample_identity(self, gender_class, ethnicity_class, num_samples=1,
                        rng=None, verbose=False):
        return self.blend_identities(
            {gender_class: 1.0}, {ethnicity_class: 1.0},
            num_samples=num_samples, rng=rng)

    def blend_identities(self, gender_weights, ethnicity_weights,
                         num_samples=1, rng=None, verbose=False):
        if not gender_weights or not ethnicity_weights:
            raise ValueError("Gender and ethnicity weights cannot be empty")
        if any(w < 0 for w in gender_weights.values()) or \
           any(w < 0 for w in ethnicity_weights.values()):
            raise ValueError("Weights cannot be negative")

        gt = sum(gender_weights.values())
        et = sum(ethnicity_weights.values())
        if np.isclose(gt, 0) or np.isclose(et, 0):
            raise ValueError("Weight sums cannot be 0")

        g_ohe = np.zeros((1, self._NUM_GENDER_CLASSES), dtype=np.float32)
        for idx, w in gender_weights.items():
            g_ohe += _one_hot([int(idx)], self._NUM_GENDER_CLASSES) * (w / gt)
        e_ohe = np.zeros((1, self._NUM_ETHNICITIES_CLASSES), dtype=np.float32)
        for idx, w in ethnicity_weights.items():
            e_ohe += _one_hot([int(idx)], self._NUM_ETHNICITIES_CLASSES) * (w / et)

        labels = np.repeat(np.concatenate([g_ohe, e_ohe], axis=1),
                           num_samples, axis=0)
        rng = _get_rng(rng)
        z = rng.normal(size=(num_samples, self._LATENT_DIM)).astype(np.float32)
        return self._decoder.predict([z, labels])


# --- Loader ------------------------------------------------------------------

EXPRESSION_MODEL = "expression_decoder_model.h5"
IDENTITY_MODEL = "identity_decoder_model.h5"


def load(data_dir):
    """data_dir: <gnm shape package>/data/semantic_sampler.
    Returns (identity_sampler, expression_sampler)."""
    ident_path = os.path.join(data_dir, IDENTITY_MODEL)
    expr_path = os.path.join(data_dir, EXPRESSION_MODEL)
    for p in (ident_path, expr_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"sampler model not found: {p}")
    return IdentitySampler(ident_path), ExpressionSampler(expr_path)
