from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {name}")
    print(" ".join(cmd))
    print(f"{'=' * 80}\n", flush=True)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the main California building baseline pipeline (Steps 2-4)."
    )
    parser.add_argument("--base-dir", default=".", help="Repository root")
    parser.add_argument("--r-home", required=True)
    parser.add_argument("--stateior-datadir", required=True)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    scripts = base / "scripts"
    inputs_public = base / "inputs" / "public"
    outputs_step2 = base / "outputs" / "step2"
    outputs_step3 = base / "outputs" / "step3"
    outputs_final = base / "outputs" / "final"

    required_files = [
        scripts / "USEEIO.py",
        scripts / "step2_build_detail_matrices.py",
        scripts / "step3_build_emissions_vectors.py",
        scripts / "step4_export_final.py",
        inputs_public / "DetailedConcordanceQuery.csv",
        inputs_public / "MRR_Emissions.xlsx",
        inputs_public / "inventory_with_new_naics.xlsx",
        inputs_public / "dnb_ca_weights.csv",
        inputs_public / "dnb_rous_weights.csv",
        inputs_public / "remi_derived_inputs.xlsx",
    ]

    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    step2_out = outputs_step2 / "ca_detailed_step2.xlsx"
    step3_out = outputs_step3 / "ca_detailed_step3_dn.xlsx"
    step4_out = outputs_final / "ca_detailed_matrices_final.xlsx"

    py = sys.executable

    run_step(
        "Step 2 - Build detailed matrices",
        [
            py,
            str(scripts / "step2_build_detail_matrices.py"),
            "--dnb-ca",
            str(inputs_public / "dnb_ca_weights.csv"),
            "--dnb-rous",
            str(inputs_public / "dnb_rous_weights.csv"),
            "--remi-input",
            str(inputs_public / "remi_derived_inputs.xlsx"),
            "--r-home",
            args.r_home,
            "--stateior-datadir",
            args.stateior_datadir,
            "--out-step2",
            str(step2_out),
        ],
    )

    run_step(
        "Step 3 - Build emissions vectors",
        [
            py,
            str(scripts / "step3_build_emissions_vectors.py"),
            "--step2-file",
            str(step2_out),
            "--inventory-file",
            str(inputs_public / "inventory_with_new_naics.xlsx"),
            "--concordance-file",
            str(inputs_public / "DetailedConcordanceQuery.csv"),
            "--dnb-ca",
            str(inputs_public / "dnb_ca_weights.csv"),
            "--dnb-rous",
            str(inputs_public / "dnb_rous_weights.csv"),
            "--r-home",
            args.r_home,
            "--stateior-datadir",
            args.stateior_datadir,
            "--out-step3",
            str(step3_out),
        ],
    )

    run_step(
        "Step 4 - Export final workbook",
        [
            py,
            str(scripts / "step4_export_final.py"),
            "--step2-file",
            str(step2_out),
            "--step3-file",
            str(step3_out),
            "--r-home",
            args.r_home,
            "--stateior-datadir",
            args.stateior_datadir,
            "--out-final",
            str(step4_out),
        ],
    )

    print("\nPipeline completed successfully.")
    print(f"Final workbook: {step4_out}")


if __name__ == "__main__":
    main()