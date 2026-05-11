"""
Subject inclusion/exclusion filter for the DMT-MED fMRI dataset.

Excluded subjects: sub-06, sub-08, sub-12, sub-26, sub-36
Criterion: any run with >50% frames censored at FD > 0.3 mm (within-subject design).
Source: 01_preprocessing/01_QC/results/excluded_subjects.tsv
Final sample: 35 subjects × 2 sessions = 70 runs.
"""

# Hard exclusions per 01_preprocessing/01_QC results — CLAUDE.md "Subject exclusion"
_EXCLUDED = frozenset(["sub-06", "sub-08", "sub-12", "sub-26", "sub-36"])


def get_included_subjects(all_subjects=None):
    """
    Return sorted list of included subjects after applying fixed exclusions.

    Excluded: sub-06, sub-08, sub-12, sub-26, sub-36 (criterion: >50% frames
    censored at FD > 0.3 mm in any run). See 01_preprocessing/01_QC/results/excluded_subjects.tsv.

    Parameters
    ----------
    all_subjects : list of str, optional
        Full subject list. Defaults to sub-01 .. sub-40 (all 40).

    Returns
    -------
    list of str — 35 included subjects in sorted order.
    """
    if all_subjects is None:
        all_subjects = [f"sub-{i:02d}" for i in range(1, 41)]
    return sorted(s for s in all_subjects if s not in _EXCLUDED)
