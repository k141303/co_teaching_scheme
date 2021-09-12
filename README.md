# Co-Teaching Student-Model through Submission Results of Shared Task
This repository is an implementation used in "Co-Teaching Student-Model through Submission Results of Shared Task."  

## Operating Environment

You can use Dockerfile to construct the operating environment.

```sh
cd docker
docker build -t [image_name] .
```

## Data

You need to download the [pre-trained BERT model](https://drive.google.com/file/d/1ASfiB1JfmsNQGm3y7P7PqX_z04eVJHKn/view?usp=sharing) and the [preprocessed dataset](https://drive.google.com/file/d/14J4KQVdZM44InrgCcc8r86mkjnQh1fk5/view?usp=sharing), and unzip these zip files and place them in the project root folder.

## Train model in "Co-Teaching" setting on Airport category

```sh
OUTPUT=./outputs/co_teaching
TASK=JP-5
CATEGORY=Airport
python3 code/knowledge_distillation.py \
    hydra.run.dir="${OUTPUT%/}/hydra" \
    log_dir="${OUTPUT%/}/tensorboard" \
    model.file_dir=model/roberta_base_wiki201221_janome_vocab_32000/ \
    model.output_dir="${OUTPUT%/}/${TASK}/${CATEGORY}" \
    data.train_dir=dataset/preprocessed_train/${TASK}/${CATEGORY} \
    data.system_results_dir=dataset/preprocessed_system_results_2000/${TASK}/${CATEGORY} \
    train.epoch=100 \
    train.early_stop=100 \
    train.batch_size=16 \
    train.class_weight_scheme=balanced_focal \
    train.balanced_loss_beta=0.99999 \
    data.max_sample_size=2000
```
## Train model in "Non-Teaching" setting on Airport category

```sh
OUTPUT=./outputs/non_teaching
TASK=JP-5
CATEGORY=Airport
python3 code/train.py \
    hydra.run.dir="${OUTPUT%/}/hydra" \
    log_dir="${OUTPUT%/}/tensorboard" \
    model.file_dir=model/roberta_base_wiki201221_janome_vocab_32000/ \
    model.output_dir="${OUTPUT%/}/${TASK}/${CATEGORY}" \
    data.train_dir=dataset/preprocessed_train/${TASK}/${CATEGORY} \
    train.epoch=100 \
    train.early_stop=100 \
    train.batch_size=16 \
    train.class_weight_scheme=balanced_focal \
    train.balanced_loss_beta=0.99999
```
