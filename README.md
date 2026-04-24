# California Building Sector Baseline Package

This repository builds a California-specific detailed two-region EEIO baseline for the building sector using USEEIO/stateio outputs, public supporting datasets, and publishable derived input files used by the modeling pipeline.

## Repository structure

```text
baseline-package/
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ .gitignore
├─ run_all.py
├─ inputs/
│  ├─ public/
│  └─ private/
├─ modelspecs/
├─ outputs/
│  ├─ step1/
│  ├─ step2/
│  ├─ step3/
│  └─ final/
└─ scripts/
   ├─ USEEIO.py
   ├─ step1_build_dnb_weights.py
   ├─ step2_build_detail_matrices.py
   ├─ step3_build_emissions_vectors.py
   └─ step4_export_final.py
```

## Main workflow vs optional preprocessing

The default runnable workflow in this repository starts from publishable derived input files and runs:

- Step 2: build detailed matrices
- Step 3: build emissions vectors
- Step 4: export the final consolidated workbook

Step 1 is kept in the repository as an optional preprocessing step used only to regenerate DnB-derived weight files from raw DnB source data. Raw DnB source data is not included in this repository.

## Required inputs

### `inputs/public/`

Place the following files in `inputs/public/`:

- `DetailedConcordanceQuery.csv`
- `MRR_Emissions.xlsx`
- `inventory_with_new_naics.xlsx`
- `dnb_ca_weights.csv`
- `dnb_rous_weights.csv`
- `remi_derived_inputs.xlsx`

### `inputs/private/`

This folder is kept only for optional internal preprocessing. It is not needed for the default Steps 2–4 workflow.

For internal use only, Step 1 may require:

- `DNB_REVENUS.xlsx`

## Outputs

The pipeline writes:

### Step 2
- `outputs/step2/ca_detailed_step2.xlsx`

### Step 3
- `outputs/step3/ca_detailed_step3_dn.xlsx`

### Step 4
- `outputs/final/ca_detailed_matrices_final.xlsx`

## Prerequisites

This workflow assumes:

- Windows
- Anaconda or Miniconda
- a local R installation
- the R packages `useeior` and `stateior`
- a valid `STATEIOR_DATADIR`
- required USEEIO/stateio model specs available locally
- `USEEIO.py` placed in the `scripts/` folder

## Python environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate buildings
```

If the environment already exists, update it instead:

```bash
conda env update -n buildings -f environment.yml --prune
conda activate buildings
```

## Python requirements

If you prefer pip for the Python side:

```bash
pip install -r requirements.txt
```

## R / stateio setup

You must also have a working local R installation and the R packages:

- `useeior`
- `stateior`

The scripts require `R_HOME` and `STATEIOR_DATADIR`.

Example values:

```text
R_HOME = C:\Users\lguillot\AppData\Local\Programs\R\R-4.4.2
STATEIOR_DATADIR = C:\Users\lguillot\AppData\Local\stateio
```

## Setup verification

Before running the pipeline, verify the environment:

```powershell
conda activate buildings
python -c "import pandas, numpy, openpyxl, rpy2; print('python deps ok')"
python -c "import sys; sys.path.insert(0, r'.\scripts'); import USEEIO; print('USEEIO ok')"
Rscript -e "library(useeior); library(stateior); cat('R packages ok\n')"
```

## Default full pipeline

Run the default Steps 2–4 workflow from the repository root:

```powershell
python run_all.py --r-home "C:\Users\lguillot\AppData\Local\Programs\R\R-4.4.2" --stateior-datadir "C:\Users\lguillot\AppData\Local\stateio"
```

## Step-by-step execution

### Step 2

Build detailed `x`, `f`, `Z`, `A`, and `L` with REMI reconciliation:

```powershell
python scripts/step2_build_detail_matrices.py --dnb-ca "inputs/public/dnb_ca_weights.csv" --dnb-rous "inputs/public/dnb_rous_weights.csv" --remi-input "inputs/public/remi_derived_inputs.xlsx" --r-home "C:\Users\lguillot\AppData\Local\Programs\R\R-4.4.2" --stateior-datadir "C:\Users\lguillot\AppData\Local\stateio" --out-step2 "outputs/step2/ca_detailed_step2.xlsx"
```

### Step 3

Build detailed direct and total emissions vectors:

```powershell
python scripts/step3_build_emissions_vectors.py --step2-file "outputs/step2/ca_detailed_step2.xlsx" --inventory-file "inputs/public/inventory_with_new_naics.xlsx" --concordance-file "inputs/public/DetailedConcordanceQuery.csv" --dnb-ca "inputs/public/dnb_ca_weights.csv" --dnb-rous "inputs/public/dnb_rous_weights.csv" --r-home "C:\Users\lguillot\AppData\Local\Programs\R\R-4.4.2" --stateior-datadir "C:\Users\lguillot\AppData\Local\stateio" --out-step3 "outputs/step3/ca_detailed_step3_dn.xlsx"
```

### Step 4

Export the final consolidated workbook:

```powershell
python scripts/step4_export_final.py --step2-file "outputs/step2/ca_detailed_step2.xlsx" --step3-file "outputs/step3/ca_detailed_step3_dn.xlsx" --r-home "C:\Users\lguillot\AppData\Local\Programs\R\R-4.4.2" --stateior-datadir "C:\Users\lguillot\AppData\Local\stateio" --out-final "outputs/final/ca_detailed_matrices_final.xlsx"
```

## Optional preprocessing: Step 1

Step 1 is not part of the default public workflow. It is used only to regenerate DnB-derived weights from raw DnB source data for internal use.

Example:

```powershell
python scripts/step1_build_dnb_weights.py --dnb-raw "inputs/private/DNB_REVENUS.xlsx" --detailed-concordance "inputs/public/DetailedConcordanceQuery.csv" --mrr-emissions "inputs/public/MRR_Emissions.xlsx" --r-home "C:\Users\lguillot\AppData\Local\Programs\R\R-4.4.2" --stateior-datadir "C:\Users\lguillot\AppData\Local\stateio" --out-ca "outputs/step1/dnb_ca_weights.csv" --out-rous "outputs/step1/dnb_rous_weights.csv"
```

## Notes

- Run the scripts step by step the first time.
- Use `run_all.py` once the environment and folder structure are confirmed working.
- The default public workflow starts from publishable derived inputs.
- Raw DnB source data is not included in the repository.
