"""Report which DLLs Windows Application Control refuses to load.

Run via `uv run python scripts/check_app_control.py`.

WHY THIS EXISTS
---------------
On Windows 11 with Smart App Control enabled, a perfectly correct, correctly
resolved `torch` wheel can still fail to import:

    OSError: [WinError 4551] An Application Control policy has blocked this
    file. Error loading "...\\torch\\lib\\shm.dll" or one of its dependencies.

WinError 4551 is `ERROR_BLOCKED_BY_APP_CONTROL_POLICY`. It is *not* a missing
dependency, a broken venv, or a CUDA problem, and re-installing will never fix
it. This script separates those cases by loading every DLL in `torch/lib`
directly with `ctypes` and reporting what actually failed.

WHAT WinError 4551 ACTUALLY MEANS — and why you should re-run before acting
---------------------------------------------------------------------------
**It means the file's cloud reputation is currently bad, not that a policy is
being applied against it.**

This was learned the hard way on 2026-10-12. This script first reported **2 of 38**
torch DLLs blocked and **2 of 69** scikit-learn modules blocked, which looked
exactly like a standing machine policy, and the conclusion drawn from it — "these
phases must run elsewhere" — was **wrong**. Re-running it later the same day, with
Smart App Control **still On and unchanged**, showed **38 of 38 and 0 of 69**. Every
file had cleared on its own.

Signature status is not the discriminator either: unsigned `c10.dll` loads fine
and signed `cudart64_13.dll` also loads, so per-file reputation is what is being
applied. Mark-of-the-Web is absent from all of them, so `Unblock-File` cannot help.

**So re-run this diagnostic before changing any machine security setting.** An hour
of waiting is cheaper than disabling Smart App Control, which is not reversible
without a Windows reset. A genuine standing policy looks different: it blocks the
same files on every run, indefinitely.

Exit codes:  0 = nothing blocked · 1 = something blocked · 2 = torch not found.
"""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

APP_CONTROL_WINERROR = 4551
BLOCKED_BY_POLICY = "blocked by Application Control"


def torch_lib_dir() -> Path:
    """Locate `torch/lib` relative to this script, not the CWD."""
    root = Path(__file__).resolve().parent.parent
    return root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"


def classify(dll: Path) -> tuple[str, str]:
    """Return `(status, detail)` for one DLL. Never raises."""
    try:
        ctypes.WinDLL(str(dll))
    except OSError as exc:
        if getattr(exc, "winerror", None) == APP_CONTROL_WINERROR:
            return BLOCKED_BY_POLICY, str(exc)
        return "failed to load", f"winerror={getattr(exc, 'winerror', '?')}: {exc}"
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not crash
        return "failed to load", f"{type(exc).__name__}: {exc}"
    return "ok", ""


def main() -> int:
    lib = torch_lib_dir()
    if not lib.is_dir():
        print(f"torch/lib not found at {lib}")
        print("Run `make setup` (or `uv sync --frozen`) first.")
        return 2

    dlls = sorted(glob.glob(os.path.join(lib, "*.dll")))
    if not dlls:
        print(f"no DLLs in {lib}")
        return 2

    results = [(Path(d), *classify(Path(d))) for d in dlls]

    blocked = [(n, s, det) for n, s, det in results if s == BLOCKED_BY_POLICY]
    other = [(n, s, det) for n, s, det in results if s not in ("ok", BLOCKED_BY_POLICY)]
    ok = [n for n, s, _ in results if s == "ok"]

    print(f"torch/lib: {lib}")
    print(f"total {len(dlls)} DLLs | ok {len(ok)} | blocked {len(blocked)} | other {len(other)}")

    cuda = [
        n.name
        for n in ok
        if "cuda" in n.name.lower()
        or "cublas" in n.name.lower()
        or "cudnn" in n.name.lower()
        or "cufft" in n.name.lower()
    ]
    if cuda:
        print(f"\nCUDA runtime DLLs that loaded fine ({len(cuda)}):")
        for name in sorted(cuda):
            print(f"  + {name}")

    if blocked:
        print(f"\n{BLOCKED_BY_POLICY.upper()} ({len(blocked)}):")
        for name, _, detail in blocked:
            print(f"  - {name.name}")
            if detail:
                print(f"      {detail}")
        print("\nNote: signature status is NOT the discriminator here.")
        print("Unsigned DLLs that load fine are listed under 'ok' above.")
        print("Smart App Control applies cloud file reputation per file.")

    if other:
        print(f"\nOTHER FAILURES ({len(other)}):")
        for name, _, detail in other:
            print(f"  ! {name.name}: {detail}")

    if blocked:
        print("  Remediation — see the module docstring. In short:")
        print("    1. RE-RUN THIS FIRST. WinError 4551 is cloud file reputation,")
        print("       not a standing policy. On 2026-10-12 2 blocked torch DLLs and")
        print("       2 blocked sklearn modules both cleared on their own within the")
        print("       same day, with Smart App Control unchanged. Waiting is cheaper")
        print("       than changing a security setting, and re-enabling Smart App")
        print("       Control requires a Windows reset.")
        print("    2. If it persists across re-runs and over days, THEN suspect a")
        print("       standing policy and consider disabling Smart App Control.")
        print("\nUnaffected phases: P1, P2, P4, P5, P6, P7, P9, P10.")
        print("Blocked phases:    P3 (embedder), P8 (local reader family B).")
        return 1

    print("\nNothing blocked. `import torch` should work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
