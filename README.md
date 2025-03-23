# Testing TIFA for my master thesis

<div align="center">
<h1>TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering</h1>

[[Yushi Hu](https://yushi-hu.github.io/)], [[Benlin Liu]()], [[Jungo Kasai](https://jungokasai.github.io/)], [[Yizhong Wang](https://homes.cs.washington.edu/~yizhongw/)], [[Mari Ostendorf](https://people.ece.uw.edu/ostendorf/)], [[Ranjay Krishna](https://www.ranjaykrishna.com/index.html)], [[Noah A. Smith](https://nasmith.github.io/)]

[[Project Page](https://tifa-benchmark.github.io/)]

<h3>What is TIFA?</h3>

Images synthesized by text-to-image models (e.g. Stable Diffusion) often do not follow the text inputs well.
TIFA is a simple tool to evaluate the fine-grained alignment between the text and the image.
This repository contains the code and models for our paper [TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering](https://arxiv.org/abs/2303.11897). This paper is also accepted to [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_TIFA_Accurate_and_Interpretable_Text-to-Image_Faithfulness_Evaluation_with_Question_Answering_ICCV_2023_paper.html). Please refer to the [project page](https://tifa-benchmark.github.io/) for a quick overview.

</div>

## Quick start

```bash
conda create --name tifa python=3.8
conda activate tifa
pip install -r requirements.txt
```

## Image evaluation

The repository expects the following structure for the folder containing the images to be evaluated:

```bash
evaluation/
│── prompt_collection1/
│   │── prompt_collection1-model_name1/
│   │   ├── 000_A bus/
│   │   ├── 001_A bus and a bench/
│   │   ├── ...
│   │── prompt_collection1-model_name2/
│   │   ├── 000_A bus/
│   │   ├── 001_A bus and a bench/
│   │   ├── ...
│   │── prompt_collection1.csv
```

where:

- `evaluation`: a folder in the root of the project
- `prompt_collection1`: a folder containing everything is needed for the evaluation of that specific prompt collection
- `prompt_collection1-model_name`: a folder containing the generated images, divided by prompt (e.g. `000_A bus`, `001_A bus and a bench` etc.) by a specific model. More than one folder (and thus more than one model) can be present, all of them will be evaluated.
- `prompt_collection1.csv`: a file containing the data about prompts and bounding boxes used to generate the images

## Configuration

In the root of the project there is a file named `config.py` containing the configuration.
In particular, the `prompt_collection` field should be the same used for the .csv file and the folder in `evaluation/` (e.g. `prompt_collection1`).
Also, the tifa_version field defines which version of TIFA should be used:

- REGULAR: only the TIFA score (text-alignment) is returned
- EXTENDED: both the TIFA score (text-alignment) and AuC (layout-score) is returned

## Citation

```bibtex
@article{hu2023tifa,
  title={TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering},
  author={Hu, Yushi and Liu, Benlin and Kasai, Jungo and Wang, Yizhong and Ostendorf, Mari and Krishna, Ranjay and Smith, Noah A},
  journal={arXiv preprint arXiv:2303.11897},
  year={2023}
}
```
