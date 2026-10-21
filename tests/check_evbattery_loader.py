"""
Safety and parsing test for the EVBattery loader.

The EVBattery snippets are Python pickle files, and unpickling can run arbitrary code. This test
checks that evbattery_loader.RestrictedUnpickler (1) refuses a pickle that would call os.system,
(2) refuses one that would call builtins.eval, and (3) still reads a snippet built exactly like the
real ones (torch.save's three legacy header pickles, then a (128, 8) array and an OrderedDict of
metadata).

Synthetic data only; no project files needed. Exits with status 1 if any check fails.
Run from the repository root:  python tests/check_evbattery_loader.py
"""
import collections
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evbattery_loader as ev

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class _CallsSystem:
    def __reduce__(self):
        return (os.system, ("echo this must never run",))


class _CallsEval:
    def __reduce__(self):
        return (eval, ("1 + 1",))


def refused(payload):
    try:
        ev.load_snippet(payload)
    except pickle.UnpicklingError:
        return True
    return False


header = b"".join(pickle.dumps(x, protocol=2) for x in (0x1950A86A20F9469CFC6C, 1001, {"protocol_version": 1001}))

print("Restricted unpickler")
check("refuses a pickle that calls os.system", refused(header + pickle.dumps(_CallsSystem(), protocol=2)))
check("refuses a pickle that calls builtins.eval", refused(header + pickle.dumps(_CallsEval(), protocol=2)))

arr = np.random.default_rng(0).normal(size=(128, 8))
meta = collections.OrderedDict(label="00", car=527, charge_segment="244", mileage=40886.2,
                               capacity=np.float64(39.03))
obj = ev.load_snippet(header + pickle.dumps((arr, meta), protocol=2))
got_arr, got_meta = ev._metadata(obj)
check("reads a snippet built like the real files", np.array_equal(got_arr, arr) and got_meta == meta)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
