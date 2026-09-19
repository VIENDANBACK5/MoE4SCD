import pandas as pd

from deadtrees_gate0.audit import AUDIT_LABELS, _gate_status


def test_gate_status_does_not_promote_unreviewed_rows():
    frame = pd.DataFrame({
        "audit_label": ["", "SINGLE_CROWN", "MULTI_CROWN_GROUP"],
        "confidence": ["", "HIGH", "HIGH"],
        "site_id": [1, 1, 2],
    })
    status = _gate_status(frame)
    assert status["n_pending"] == 1
    assert status["n_verified_high_single_crown"] == 1
    assert status["counts"]["SINGLE_CROWN"] == 1


def test_audit_categories_are_exact_and_mutually_exclusive_names():
    assert len(AUDIT_LABELS) == 7
    assert len(set(AUDIT_LABELS)) == len(AUDIT_LABELS)
