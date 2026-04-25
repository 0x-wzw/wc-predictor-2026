#!/usr/bin/env python3
"""
WC2026 Master Orchestrator
===========================
Commands:
  daily        - Full daily run: ingest → compare → signals → report
  update_model - Fine-tune from latest external data
  full_report  - Generate comprehensive status + signals
  init         - Bootstrap pipeline from scratch
"""

import subprocess, sys, os
from datetime import datetime

SCRIPTS = os.path.expanduser("~/.hermes/scripts")

def run(cmd):
    print(f"\n▶ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=SCRIPTS)
    return result.returncode == 0

def daily():
    print("=" * 60)
    print(f"WC2026 DAILY PIPELINE — {datetime.now().isoformat()}")
    print("=" * 60)

    # Step 1: Ingest external odds
    if not run("python3 wc2026_unified_pipeline.py ingest"):
        print("FAILED: ingest")
        return False

    # Step 2: Compare model vs market
    if not run("python3 wc2026_unified_pipeline.py compare"):
        print("FAILED: compare")
        return False

    # Step 3: Generate signals
    if not run("python3 wc2026_unified_pipeline.py signals"):
        print("FAILED: signals")
        return False

    print("\n" + "=" * 60)
    print("✓ Daily pipeline complete")
    print("=" * 60)
    return True

def update_model():
    """Placeholder: would trigger odds-based nudge for all teams."""
    print("Fine-tune from odds disagreement...")
    # This would run nudge_from_odds for each team with significant edge
    # For now, manual: wc2026_finetune.py nudge --team X --market Y --model Z
    print("Manual: use wc2026_finetune.py nudge --team T --market P1 --model P2")
    return True

def full_report():
    return run("python3 wc2026_unified_pipeline.py report")

def init():
    print("Bootstrapping WC2026 pipeline...")
    run("python3 wc2026_finetune.py status")
    run("python3 wc2026_unified_pipeline.py status")
    print("\n✓ Pipeline initialized")
    print("\nNext steps:")
    print("  1. python3 wc2026_master.py daily      # Daily run")
    print("  2. python3 wc2026_finetune.py result --match 'TeamA:2-1:TeamB'  # After match")
    print("  3. python3 wc2026_master.py full_report  # Full status")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('daily', help='Run full daily pipeline')
    sub.add_parser('update_model', help='Fine-tune from external data')
    sub.add_parser('full_report', help='Comprehensive report')
    sub.add_parser('init', help='Bootstrap pipeline')
    args = parser.parse_args()

    cmds = {
        'daily': daily,
        'update_model': update_model,
        'full_report': full_report,
        'init': init,
    }

    if args.cmd in cmds:
        cmds[args.cmd]()
    else:
        parser.print_help()
        print("\nTypical workflow:")
        print("  wc2026_master.py init          # First time")
        print("  wc2026_master.py daily         # Every day")
        print("  wc2026_finetune.py result ...  # After each match")

if __name__ == "__main__":
    main()
