import os
import hydra

import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModel,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tensorboardX import SummaryWriter

from data.datasets import load_shinra_self_training_dataset
from model import BertForIOB2Tagging, BertForDistillingIOB2Tagging
from model.loss_fct import ClassBalancedFocalCELoss

def train(cfg, dataset, dev_dataset, model, writer):
    model.train()
    dataloader = DataLoader(
        dataset,
        shuffle=cfg.data.shuffle,
        batch_size=cfg.train.batch_size,
        num_workers = cfg.data.num_workers
    )
    total_steps = len(dataloader) * cfg.train.epoch
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.adam_B1, cfg.optim.adam_B2),
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.train.warmup_steps*total_steps,
        num_training_steps=total_steps
    )
    if cfg.device.fp16 and not cfg.device.no_cuda:
        try:
            import apex
            from apex import amp
        except ModuleNotFoundError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=cfg.device.fp16_opt_level)

    if cfg.device.n_gpu != 1:
        model = torch.nn.DataParallel(model)

    if cfg.train.class_weight_scheme == "balanced_focal":
        loss_fct = ClassBalancedFocalCELoss(
            dataset.class_counts,
            3,
            beta=cfg.train.balanced_loss_beta,
            gamma=cfg.train.focal_loss_gamma
        )
    elif cfg.train.class_weight_scheme in ["no_weight", "uniform_weight"]:
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 500.0, 500.0]).to(cfg.device.device))
    else:
         raise NotImplementedError()

    steps = 0
    best_score = None
    step_loss = []
    early_stopping_count = 0
    for epoch in range(1,cfg.train.epoch+1):
        train_outputs = []
        for batch in tqdm.tqdm(dataloader, desc=f"TRAIN {epoch}"):
            _, logits = model(
                input_ids=batch["input_ids"].to(cfg.device.device),
                attention_mask=batch["attention_mask"].to(cfg.device.device)
            )

            labels = batch["labels"].view(-1).to(cfg.device.device)
            active_logits = logits.view(-1, 3)[labels >= 0]
            active_labels = labels[labels >= 0]

            loss = loss_fct(active_logits, active_labels)

            outputs = logits.argmax(dim=-1)
            train_outputs += [*zip(
                batch["pageid"],
                batch["part_id"].tolist(),
                batch["offsets"].tolist(),
                outputs.cpu().tolist()
            )]

            step_loss.append(loss.item())

            steps += 1
            if steps % cfg.train.gradient_accumulation_steps == 0:
                if cfg.device.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                writer.add_scalar("Loss", sum(step_loss)/len(step_loss), steps)
                step_loss = []

        score = dataset.labeled_dataset.evaluate(train_outputs)
        micro_ave_score = score.pop("micro_ave")

        dev_score = eval(cfg, dev_dataset, model)
        micro_ave_dev_score = dev_score.pop("micro_ave")

        writer.add_scalars("Score__Train__MicroAve", micro_ave_score, steps)
        writer.add_scalars("Score__Dev__MicroAve", micro_ave_dev_score, steps)

        print(micro_ave_dev_score)
        if best_score is None or micro_ave_dev_score["F1"] >= best_score["F1"]:
            best_score = micro_ave_dev_score
            best_model_param = model.state_dict()

            if cfg.model.output_dir is not None:
                output_dir = hydra.utils.to_absolute_path(cfg.model.output_dir)
                print(f"Save model to {output_dir}")
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save(
                    output_dir,
                    attributes=dataset.attributes,
                    class_weight_scheme=cfg.train.class_weight_scheme
                )

            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count > cfg.train.early_stop:
            break

def eval(cfg, dataset, model):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=cfg.eval.batch_size, num_workers = cfg.data.num_workers)

    eval_outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="EVAL"):
            _, outputs = model(
                input_ids=batch["input_ids"].to(cfg.device.device),
                attention_mask=batch["attention_mask"].to(cfg.device.device)
            )
            outputs = outputs.argmax(dim=-1)
            eval_outputs += [*zip(
                batch["pageid"],
                batch["part_id"].tolist(),
                batch["offsets"].tolist(),
                outputs.cpu().tolist()
            )]

    model.train()
    return dataset.evaluate(eval_outputs)

@hydra.main(config_path="../config", config_name="self_training_config")
def main(cfg):
    cfg.device.n_gpu = torch.cuda.device_count()
    cfg.device.device = "cuda" if torch.cuda.is_available() and not cfg.device.no_cuda else "cpu"
    set_seed(cfg.seed)
    train_dataset, dev_dataset = load_shinra_self_training_dataset(cfg)

    if cfg.model.output_dir is not None:
        output_dir = hydra.utils.to_absolute_path(cfg.model.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        dev_dataset.save_pageids(os.path.join(output_dir, "dev_pageids.json"))

    import sys
    sys.exit()

    model_dir = hydra.utils.to_absolute_path(cfg.model.file_dir)
    config = AutoConfig.from_pretrained(
        os.path.join(model_dir, cfg.model.config_name),
        num_labels=len(train_dataset.attributes),
        add_pooling_layer=False,
        hidden_dropout_prob=cfg.train.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.train.attention_probs_dropout_prob
    )
    bert = AutoModel.from_pretrained(
        os.path.join(model_dir, cfg.model.model_name),
        config=config
    )
    model = BertForIOB2Tagging(
        config,
        bert,
        class_counts=train_dataset.class_counts,
        class_weight_scheme=cfg.train.class_weight_scheme,
        focal_loss_gamma=cfg.train.focal_loss_gamma,
        balanced_loss_beta=cfg.train.balanced_loss_beta,
        freeze_share_model=cfg.model.freeze_share_model
    )

    model.to(cfg.device.device)

    # TensorboardX
    log_dir = hydra.utils.to_absolute_path(cfg.log_dir)
    writer = SummaryWriter(log_dir)

    train(cfg, train_dataset, dev_dataset, model, writer)

if __name__ == '__main__':
    main()
