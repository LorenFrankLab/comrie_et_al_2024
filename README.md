## Code related to Comrie et al. 2024 Hippocampal representations of alternative possibilities are flexibly generated to meet cognitive demands
This repo contains analysis and figure code for the manuscript. All data are on the DANDI Archive.

## Cite
Comrie A.E., Monroe E.J., Kahn A.E., Denovellis E.L., Joshi A., Guidera J.A., Krausz T.A., Berke J.D., Daw N.D., Frank L.M. (2024) *Hippocampal representations of alternative possibilities are flexibly generated to meet cognitive demands.* bioRxiv 2024.09.23.613567. https://doi.org/10.1101/2024.09.23.613567

## Data
[https://dandiarchive.org/001942](https://dandiarchive.org/dandiset/001942)
Note that raw and intermediate analysis files are on DANDI in standardized NWB format so users don't need to do any heavy recomputing.
dandiset_id: 001942

## Running notebooks/making figures

#### Figure 1
- `f1bd.ipynb` — B, D
- `f1ef_f5b_sf5efgh.ipynb` — E, F

#### Figure 2
- `f2a_3a_4a_2d_decoding_exs.ipynb` — A
- `f23cde_sf2ab.ipynb` — C, D, E
- `f2f_3f.ipynb` — F

#### Figure 3
- `f2a_3a_4a_2d_decoding_exs.ipynb` — A
- `f23cde_sf2ab.ipynb` — C, D, E
- `f2f_3f.ipynb` — F

#### Figure 4
- `f2a_3a_4a_2d_decoding_exs.ipynb` — A
- `f4cdefg_sf4ab.ipynb` — C, D, E, F, G

#### Figure 5
- `f1ef_f5b_sf5efgh.ipynb` — B
- `f4cdefg_sf4ab.ipynb` — A (combines Fig. 4 D, G)

#### Supplemental Figure 1
- `sf1a.ipynb` — A
- `sf1ef.ipynb` — E, F

#### Supplemental Figure 2
- `f23cde_sf2ab.ipynb` — A, B
- `sf2cd_theta.ipynb` — C, D
- `sf2cd_theta_stats.ipynb` — C, D (stats)

#### Supplemental Figure 3
- `f23cde_sf2ab.ipynb` — set parameters per panel
- `f4cdefg_sf4ab.ipynb` — set parameters per panel

#### Supplemental Figure 4
- `f4cdefg_sf4ab.ipynb` — A, B
- `sf4cde.ipynb` — C, D, E

#### Supplemental Figure 5
- `sf5_abcd.ipynb` — A, B, C, D
- `f1ef_f5b_sf5efgh.ipynb` — E, F, G, H

## Overview
- Reproduce figures via the Docker image (bundles database and environment), or use the code with your own Spyglass/DataJoint setup
- Install this package (`pip install -e .`)
- Run the reconstruction notebook once to populate `data/`
- Run any notebook, make figures, explore

> **Easiest path:** use the Docker image (see *Docker*) which includes the database, environment, and notebooks, and streams data from DANDI for you. Alternatively the manual setup info below is for running outside Docker.

## Repository structure
```
src/         analysis and plotting modules (installed via `pip install -e .`)
  paths.py   central data/figure path configuration
notebooks/   figure notebooks and initial data notebook
env/         conda info
data/        (gitignored) reconstructed analysis files — created by initial notebook
figures/     (gitignored) any saved out panels
```

## Docker (recommended, WIP)
A Docker image bundling database, environment, and notebooks together is the simplest way to reproduce the figures. Note work in progress.

## Manual installation
Requires `conda`/`mamba` and Python ≥ 3.9.
```bash
# 1. Create the analysis environment
conda env create -f env/conda_env.yml
conda activate <env-name> # name at top of env .yml

# 2. Install this repo's modules
pip install -e .
```

#### Connect to database
The notebooks query a DataJoint/MySQL database. The Docker image provides this prepopulated. Or, for advanced use, run code with your own working and configured Spyglass environment (docs here: https://lorenfranklab.github.io/spyglass/). Once connected, NWB files will be streamed from DANDI.

## Configuration (optional)
Note by default reconstructed data go in `./data/` and figures in `./figures/`. You can override with environment variables, for ex in Docker if you want to keep data files elsewhere:
```bash
export SPATIAL_BANDIT_DATA_DIR=/path/to/data
export SPATIAL_BANDIT_FIG_DIR=/path/to/figures
```
Paths are for the most part resolved centrally in `src/paths.py`

## Data preparation
Note the code here uses two data sources, all from the same DANDI dataset:
1. **Spyglass database queries** stream NWB files from DANDI on demand (no local raw files needed).
2. **Additional analysis files** a subset of files not initially stored in NWB format are reconstructed into a local `data/` folder by a one-time notebook **`notebooks/reconstruct_additional_data_files.ipynb`** that streams NWBs from DANDI into `data/`.
