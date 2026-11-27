"""Report which DLLs Windows Application Control refuses to load."""

from __future__ import annotations

import ctypes
import glob
import os
from pathlib import Path

APP_CONTROL_WINERROR = 4551
BLOCKED_BY_POLICY = "blocked by Application Control"


def torch_lib_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"


def classify(dll: Path) -> tuple[str, str]:
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
