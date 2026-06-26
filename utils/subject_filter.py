"""
Subject inclusion/exclusion filter for the DMT-MED fMRI dataset.

Excluded subjects: sub-06, sub-08, sub-12, sub-26, sub-36
Criterion: any run with >50% frames censored at FD > 0.3 mm (within-subject design).
Source: 99_QC/01_motion_qc/results/excluded_subjects.tsv
Final sample: 35 subjects × 2 sessions = 70 runs.
"""

# Hard exclusions per 99_QC/01_motion_qc results — CLAUDE.md "Subject exclusion"
_EXCLUDED = frozenset(["sub-06", "sub-08", "sub-12", "sub-26", "sub-36"])


def get_included_subjects(all_subjects=None):
    """
    Return sorted list of included subjects after applying fixed exclusions.

    Excluded: sub-06, sub-08, sub-12, sub-26, sub-36 (criterion: >50% frames
    censored at FD > 0.3 mm in any run). See 99_QC/01_motion_qc/results/excluded_subjects.tsv.

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


# Only sub-12 is unusable (broken/missing run); the other four high-motion
# subjects are KEPT at this stage so the denoising-robustness pipelines all share
# one sample. Motion is handled later as a stats-stage covariate, not by exclusion.
_PIPELINE_EXCLUDED = frozenset(["sub-12"])


def get_pipeline_subjects(all_subjects=None):
    """
    Return sorted subject list for the three-pipeline denoising robustness analysis.

    n = 39: all subjects minus sub-12 only. Unlike get_included_subjects() (n=35),
    this KEEPS the 4 high-motion subjects (sub-06, sub-08, sub-26, sub-36) so the
    detrend / glm / maximal pipelines run on an identical sample. High motion is
    carried into the statistics stage as a covariate (mean FD, percent-censored),
    not removed here.

    Do NOT use this for the legacy n=35 analysis — use get_included_subjects().

    Returns
    -------
    list of str — 39 subjects in sorted order.
    """
    if all_subjects is None:
        all_subjects = [f"sub-{i:02d}" for i in range(1, 41)]
    return sorted(s for s in all_subjects if s not in _PIPELINE_EXCLUDED)
