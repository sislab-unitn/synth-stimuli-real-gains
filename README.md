# Synthetic Stimuli, Real Gains: Rethinking VLM Fine-Tuning Through Fully Controlled Data Generation

## Abstract

Performance gains of Vision Language Models (VLMs) obtained by fine-tuning are generally based on ad hoc data collection and annotation of real-world scenes. Despite the improvements, this process is often prone to biases, errors, and distribution imbalance, resulting in overfitting and imbalanced performance. Although a few studies have explored synthetic data generation, they typically lack control over data distribution and annotation quality. In this work, we re-evaluate the potential of model fine-tuning by exploring a fully controlled data generation and annotation pipeline, obtaining bias-free data with balanced distribution and clean annotations. Using the spatial reasoning task of identifying the absolute position of an object as a use case, we fine-tune state-of-the-art VLMs and conduct exhaustive evaluations on both synthetic and real-world benchmarks, including transferability to real-world scenes. Our experiments reveal two key findings: 1) fine-tuning on balanced data yields uniform performance across the visual scene and mitigates common biases with as few as 130 samples; and 2) fine-tuning on synthetic stimuli improves performance by 13% on real-world data (COCO), outperforming models fine-tuned on the full COCO train set.

# Usage

## Installation

Python 3.10.16 was used for the experiments.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r world/requirements.txt
```

Note: `llava-onevision-7b` needs `transformers>=4.53`, while `requirements.txt` pins `transformers==4.46.3`.

## Data generation

We used [CIVET](https://aclanthology.org/2025.findings-emnlp.239/) to generate the synthetic data. Please cite the [original work](#civet-citation) when generating new data.

All commands are run from the repository root, as modules (`python -m world.<script>`). The first argument is the output folder.

Synthetic train set:

```
python -m world.generate_civet_exhaustive ./data/civet_exhaustive --img_size 672
```

Synthetic test set:

```
python -m world.generate_civet_test ./data/civet_test --img_size 672
```

Real-world (COCO) test set. First download COCO val2017 and its annotations (saved to `coco/`):

```
./download_coco.sh
```

then generate the test set:

```
python -m world.generate_coco_single_inst ./data/coco_test
```

## Fine-tuning

Run from the repository root. Example: fine-tuning `molmo-o-7b` on the `./data/civet_exhaustive` set (synthetic train set) and testing on both the synthetic and COCO test sets:

```
python -m main molmo-o-7b ./data/civet_exhaustive ft_civet_exhaustive_5_runs fine-tune --batch-size 1 --val-batch-size 1 --runs 5 --test_sets data/civet_test/ data/coco_test/
```

Available models: `molmo-o-7b`, `llava-next-7b`, `llava-onevision-7b`, `qwen2-vl-7b`, `CLIP`. Results and the run arguments are saved in `output/<MODEL_NAME>/<EXPERIMENT_NAME>/`. Add `--test_only` to skip training and only evaluate on the given `--test_sets`. See `python -m main <MODEL_NAME> <DATA_FOLDER> <EXPERIMENT_NAME> fine-tune --help` for all options.

## Analysis

Run from the repository root. Example: computing the results of the `molmo-o-7b` fine-tuning experiment above on both the synthetic and COCO test sets:

```
python -m analysis --models molmo-o-7b --experiments ft_civet_exhaustive_5_runs --test_sets civet_test coco_test --save_results
```

Test sets are given by folder name inside `--data_folder` (default `./data`), and results are read from `--results_folder` (default `./output`). For experiments with multiple runs, results are averaged over runs. Multiple models and experiments can be passed at once (e.g., `--experiments baseline ft_civet_exhaustive_5_runs`). With `--save_results`, tables (accuracy, answer lengths, % of "other" answers) are saved as CSV in `output/`, and plots in `analysis/`.

# How to cite

```bibtex
@article{
    rizzoli2026syntheticstimulirealgains,
    title={Synthetic Stimuli, Real Gains: Rethinking VLM Fine-Tuning Through Fully Controlled Data Generation},
    author={Massimo Rizzoli and Simone Alghisi and Seyed Mahed Mousavi and Giuseppe Riccardi},
    year={2026},
    eprint={2511.11440},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2511.11440},
}```
