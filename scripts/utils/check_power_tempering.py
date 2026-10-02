import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.power_tempering import mh_log_accept, swap_log_accept


def kernel(alpha):
    states = [(0,), (1, 0), (1, 1)]
    base = np.array([0.5, 0.25, 0.25])
    target = base ** alpha
    target /= target.sum()
    matrix = np.zeros((3, 3))
    for i, old in enumerate(states):
        for restart in [0, 1]:
            if restart >= len(old):
                matrix[i, i] += 0.5
                continue
            eligible = [j for j, new in enumerate(states) if new[:restart] == old[:restart]]
            proposal = {j: (0.5 if len(states[j]) == 1 else 0.5 ** (len(states[j]) - restart)) for j in eligible}
            assert abs(sum(proposal.values()) - 1) < 1e-12
            for j, probability in proposal.items():
                reverse = 0.5 if len(old) == 1 else 0.5 ** (len(old) - restart)
                acceptance = math.exp(mh_log_accept(math.log(base[i]), math.log(base[j]), alpha,
                                                     math.log(reverse), math.log(probability)))
                matrix[i, j] += 0.5 * probability * acceptance
                matrix[i, i] += 0.5 * probability * (1 - acceptance)
    assert np.allclose(matrix.sum(1), 1, rtol=0, atol=1e-12)
    assert np.allclose(target @ matrix, target, rtol=0, atol=1e-12)
    return base, target, matrix


def main():
    base, t1, m1 = kernel(1.0)
    _, t2, m2 = kernel(2.0)
    np.testing.assert_allclose(t2, [2/3, 1/6, 1/6], rtol=0, atol=1e-12)
    pairs = list(itertools.product(range(3), repeat=2))
    swap = np.zeros((9, 9))
    for i, (a, b) in enumerate(pairs):
        j = pairs.index((b, a))
        acceptance = math.exp(swap_log_accept(math.log(base[a]), math.log(base[b]), 1.0, 2.0))
        swap[i, j] += acceptance
        swap[i, i] += 1 - acceptance
    joint = np.kron(t1, t2)
    transition = np.kron(m1, m2) @ swap
    np.testing.assert_allclose(joint @ transition, joint, rtol=0, atol=1e-12)
    assert transition[pairs.index((0, 0)), pairs.index((2, 2))] > 0
    print(json.dumps(dict(status='passed', exact_power_target=t2.tolist(), mh_target_preserved=True,
                         replica_joint_target_preserved=True, early_eos_can_lengthen=True, training_updates=0)))


if __name__ == '__main__':
    main()
