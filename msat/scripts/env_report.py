"""Print the resolved environment and GPU visibility."""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _version(mod_name: str) -> str:
    import importlib.metadata as md

    mod = __import__(mod_name)
    ver = getattr(mod, "__version__", None)
    if ver:
        return str(ver)
    try:
        return md.version(mod_name.replace("_", "-"))
    except Exception:  # noqa: BLE001 - best-effort reporting
        return "n/a"


def main() -> int:
    _section("interpreter")
    print(f"python      {sys.version.split()[0]}")
    print(f"platform    {platform.platform()}")
    print(f"machine     {platform.processor() or 'n/a'}")

    _section("core stack")
    mods = [
        "msat",
        "torch",
        "transformers",
        "sentence_transformers",
        "duckdb",
        "pyarrow",
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
        "plotly",
        "kaleido",
        "ijson",
        "pydantic",
    ]
    for name in mods:
        try:
            print(f"  {name:22s} {_version(name)}")
        except ImportError:
            print(f"  {name:22s} MISSING")
        except OSError as exc:
            print(
                f"  {name:22s} BLOCKED by Application Control (WinError "
                f"{getattr(exc, 'winerror', '?')})"
            )
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            print(f"  {name:22s} ERROR ({type(exc).__name__}: {exc})")

    _section("torch / CUDA")
    try:
        import torch

        print(f"torch          {torch.__version__}")
        built_cuda = torch.version.cuda
        print(f"built with CUDA {built_cuda or 'NONE — CPU-ONLY WHEEL'}")
        print(f"cuda available  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                cap = f"{p.major}.{p.minor}"
                print(
                    f"  gpu[{i}] {p.name}  sm_{cap.replace('.', '')}  "
                    f"{p.total_memory / 1024**3:.1f} GB"
                )
            print(f"  arch list    {torch.cuda.get_arch_list()}")
        else:
            print("  -> no CUDA device. Either the CPU wheel was installed, or")
            print("     the NVIDIA driver predates CUDA 13.2 (needs >= R595).")
    except ImportError:
        print("  torch not installed — run `make setup`")
    except OSError as exc:
        print("  torch INSTALLED BUT BLOCKED BY WINDOWS APPLICATION CONTROL")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("  This is a machine security policy, not a dependency problem.")
        print("  Phases NOT blocked: P1, P2, P4, P5, P6, P7, P9, P10 — none import torch.")
        print("  Phases blocked:     P3 (embedder) and P8 (local reader family B).")
        print()
        print("  List exactly which DLLs are refused:")
        print("    uv run python scripts/check_app_control.py")
        print()
        print("  Remediation, in order of preference:")
        print("    1. Turn OFF Smart App Control, then reboot:")
        print("       Settings > Privacy & security > Windows Security >")
        print("       App & browser control > Smart App Control > Off")
        print("    2. Retry later — Smart App Control uses cloud file reputation,")
        print("       which improves as a widely-downloaded file is seen more often.")
        print("    3. Run the GPU phases under WSL2 / Linux.")

    _section("figures")
    try:
        print(f"plotly         {_version('plotly')}")
        print(f"kaleido        {_version('kaleido')} (needs Chrome for static export)")
        chrome_found = any(
            Path(p).exists()
            for p in (
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            )
        )
        print(f"chrome         {'found' if chrome_found else 'NOT FOUND'}")
        if not chrome_found:
            print("  -> run `make chrome` before `make figures`")
    except ImportError:
        print("  plotly/kaleido not installed")
    except OSError as exc:
        print(f"  plotly/kaleido present but BLOCKED: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
