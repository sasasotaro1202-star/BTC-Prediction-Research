import numpy as np
from src.time_context_correction_oos import _meta_features, _time_features

def _rows():
    return [
        {"created":"2026-09-21T00:00:00+00:00"},
        {"created":"2026-09-21T06:00:00+00:00"},
        {"created":"2026-09-28T00:00:00+00:00"},
    ]

def test_time_features_are_finite_and_periodic():
    x=_time_features(_rows())
    assert x.shape==(3,4)
    assert np.isfinite(x).all()
    assert np.allclose(x[0],x[2])

def test_meta_features_shape():
    rows=_rows()
    p=np.tile([0.2,0.3,0.5],(3,1))
    x=_meta_features(rows,p)
    assert x.shape==(3,7)
    assert np.isfinite(x).all()
