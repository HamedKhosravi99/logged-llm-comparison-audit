"""The two labelled oracle controls that bracket what a randomized audit can and cannot repair.

oracle_selection_corrected_surface  removes ALL router-induced selection bias but keeps the
                                    operational evaluator: it is the common-population operational
                                    surface mu^op_kj = E[Ytilde^o_i(k) | Jctx=j]. This is the ceiling
                                    for a perfect assignment-validity repair.
reference_outcome_oracle_surface    the designated target surface mu^ref_kj itself. The gap between
                                    the two is the measurement-validity floor: the part of the
                                    deployment loss that no amount of randomized auditing removes.

Both read the scoring cohort's outcomes and are therefore available ONLY to these explicitly
labelled controls (enforced by `validation.faithfulness_audit`).
"""


def oracle_selection_corrected_surface(surf):
    """mu^op: assignment invalidity removed, evaluator distortion intentionally retained."""
    return surf["Ef_op"].copy()


def reference_outcome_oracle_surface(surf):
    """mu^ref: the designated target outcome. Not a claim of universal real-world ground truth."""
    return surf["Ef_ref"].copy()
