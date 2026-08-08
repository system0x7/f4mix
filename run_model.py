"""Single-file F4Mix run definition.

Edit this file and run it with python run_model.py (with f4mix
installed, or from this directory).
"""

from pathlib import Path

import f4mix as fm


DATA_PREFIX = Path(
    "/path/to/dataset_prefix"
)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "runs" / "example"

TARGET_POPULATIONS = (
    "Icelandic",
    "Orcadian",
    #"Basque",
    "Sardinian",
    #"Italian_South",
    "Sicilian",
    "Greek",
    "Czechia_EBA_CordedWare",
)

SOURCE_POPULATIONS = (
    "Turkey_Barcin_Neolithic-DG",
    "Russia_Samara_EBA_Yamnaya",
    "Luxembourg_Loschbour_Mesolithic",  
    "Morocco_EN",
    "Georgia_KotiasKlde_Mesolithic",
    "Iran_GanjDareh_N",
    "Jordan_PPNB",
)
OUTGROUP = "Chimp"
RIGHT_POPULATIONS = (
    "Mbuti",
    "Switzerland_Epipaleolithic",
    "Russia_Vologda_Mesolithic",
    "Ethiopia_MotaCave_4500BP",
    "Iran_BeltCave_Mesolithic",
    "Georgia_Satsurblia_LateUP",
    "Natufian",
    "Morocco_Iberomaurusian",
    "Turkey_PPN",
)

# Czechia_EBA_CordedWare samples below 50,000 effective f4 SNPs in the
# v66_ho panel. Exclusions are applied before any genotype data are read.
EXCLUDE_SAMPLES: tuple[str, ...] = (
    "VLI075.AG",
    "TRM001.AG",
    "KON005.AG",
    "KON003.AG",
    "OBR002.AG",
    "VLI015.AG",
    "VLI081.AG",
    "OBR001.AG",
    "VLI088.AG",
    "BUT002.AG",
    "VLI085.AG",
    "VLI070.AG",
    "VLI090.AG",
    "ZEL001.AG",
    "KON001.AG",
    "OBR003.AG",
    "OBR004.AG",
    "BUT003.AG",
    "VLI019.AG",
)

if OUTPUT_DIRECTORY.exists():
    raise SystemExit(
        f"Output directory already exists: {OUTPUT_DIRECTORY}\n"
        "Choose a new output directory before starting another fit."
    )

LOAD_POPULATIONS = tuple(
    dict.fromkeys(
        (
            *TARGET_POPULATIONS,
            *SOURCE_POPULATIONS,
            OUTGROUP,
            *RIGHT_POPULATIONS,
        )
    )
)

data = (
    fm.open_genotypes(DATA_PREFIX)
    .select(populations=LOAD_POPULATIONS)
    .exclude(samples=EXCLUDE_SAMPLES)
)

model = fm.F4Model(
    sources=SOURCE_POPULATIONS,
    right=RIGHT_POPULATIONS,
    outgroup=OUTGROUP,
    blgsize=0.05,
    chunk_size=250_000,
    covariance_ridge=1e-5,
    verbose=True,
)

result = model.fit(data, target_populations=TARGET_POPULATIONS)
result.save(OUTPUT_DIRECTORY)

print(result.summary_frame())
