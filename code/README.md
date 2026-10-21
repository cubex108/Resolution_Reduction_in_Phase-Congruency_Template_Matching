# Running the research code

Use Python 3.10 and the versions in `requirements.txt` as the recorded reference environment. This list is not a complete OS lock. From the repository root, use a separate environment and install dependencies with `python -m pip install -r code/requirements.txt`.

## Reanalyse saved numerical observations

Run these commands in a COPY of the repository. They recompute derived statistics and do not rerun image matching. The Python and stress summary scripts write into their own experiment folders.

```text
python code/summarize_results.py --input results/main_rows.jsonl --out-dir reanalysis/primary --bootstrap 10000 --correlation-bootstrap 2000 --seed 12345
python code/summarize_controls.py --results-dir results --out-dir reanalysis/controls --bootstrap 10000 --seed 12345
python experiments/python_baselines/summarize_python.py
python experiments/speckle_stress/summarize_stress.py
```

The delivered observations are the frozen numerical results used by the manuscript. Field definitions and sampling units are indexed in `notes/source_data_dictionary_20261018.txt`.

## Numerical implementation checks

From `code/`, run `python -m unittest test_imaging_core test_matching_metrics -v`. From `experiments/python_baselines/`, run `python validate_python.py` in a working copy. The delivered `validation.json` records the original 19 numerical/coordinate checks. See `IMPLEMENTATION.md` in that folder for exact algorithm choices and limits. The stress-model tests are in `experiments/speckle_stress/test_stress.py`.

## Original image access and full execution

QXS-SAROPT official project: https://github.com/yaoxu008/QXS-SAROPT
ISPRS Potsdam: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

The authors confirmed the SAR source as official QXS-SAROPT. The historical local folder name below is retained by the frozen runner. Obtain original provider files under their access terms; this repository contains no raw images or NPZ pixel caches. The runner expects the following relative layout, with the code repository and `Code_论文` as siblings.

```text
study/
  repository/
    code/
    results/
  Code_论文/func/data/
    QXSLAB_SAROPT/QXSLAB_SAROPT/
      opt_256_oc_0.2/*.png
      sar_256_oc_0.2/*.png
    RGB_Lable/*_RGB.tif
    Lable_RGB/*_label.tif
    RGB_IR/*_RGB.tif
    IR_RGB/*_RGBIR.tif
```

The `Lable` spelling is intentional. RGBIR targets use the fourth, NIR-only band. Pair selections, file hashes, crop coordinates and channel choices are in `results/protocol.json` and `results/crops/*.json`.

For a NEW execution, start from a separate copy of the code with an empty `results` directory. Run `python code/run_benchmark.py prepare`, then `freeze`, `main --workers 3`, `controls --workers 3`, and `timing`. Run timing alone, without competing compute jobs. Preserve the delivered observations elsewhere. Freezing in the new environment creates a new execution signature; never edit an old protocol to accept new source or machine fingerprints.

Additional experiment sequences are documented in their folders. Their matching stages require the prepared crop caches. Numerical reanalysis uses saved rows and requires the unchanged frozen source hashes, but not the original machine path. The primary comparison is 100 SAR, 12 label and ten NIR pairs with six repeated offsets. Pair bootstrap is conditional on the stated sampling assumptions; parent-scene/spatial independence is not established.

## Figures

The paper uses native OriginPro 2026 plots. All reported figure values are derived from the observations and statistics supplied under `results/` and `experiments/`. Image and map panels additionally require the provider images, which are not redistributed here. Historical manuscript-production, figure-source and correspondence scripts are excluded from this research-code release.
