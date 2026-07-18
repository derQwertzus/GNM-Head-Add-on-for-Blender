"""Compare the NumPy sampler backend against the original TensorFlow one.

Run inside an environment that has BOTH gnm (with TF) and h5py, e.g. your
original gnm-env:

    python verify_semantic.py

Same seeds through both backends must produce (near-)identical vectors; max
abs differences should be ~1e-5 or smaller (float32 noise).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy_sampler as ns  # noqa: E402

from gnm.shape import semantic_sampler as ss  # noqa: E402
import gnm.shape.gnm_numpy as gn  # noqa: E402

data_dir = os.path.join(os.path.dirname(gn.__file__), "data", "semantic_sampler")
np_ident, np_expr = ns.load(data_dir)
tf_ident, tf_expr = ss.IdentitySampler(), ss.ExpressionSampler()

SEED = 1234

# Identity blend
gw = {ss.Gender.FEMALE: 0.3, ss.Gender.MALE: 0.7}
ew = {ss.Ethnicity.ASIAN: 0.5, ss.Ethnicity.BLACK: 0.5}
a = tf_ident.blend_identities(gw, ew, num_samples=2,
                              rng=np.random.default_rng(SEED))
b = np_ident.blend_identities(
    {ns.Gender.FEMALE: 0.3, ns.Gender.MALE: 0.7},
    {ns.Ethnicity.ASIAN: 0.5, ns.Ethnicity.BLACK: 0.5},
    num_samples=2, rng=np.random.default_rng(SEED))
print(f"identity blend  max|diff| = {np.abs(a - b).max():.3e}")

# Expression sample
a = tf_expr.sample_expression(ss.Expression.HAPPY, num_samples=2,
                              rng=np.random.default_rng(SEED))
b = np_expr.sample_expression(ns.Expression.HAPPY, num_samples=2,
                              rng=np.random.default_rng(SEED))
print(f"expression      max|diff| = {np.abs(a - b).max():.3e}")

print("OK — differences at float32 noise level mean the backends are equivalent.")
