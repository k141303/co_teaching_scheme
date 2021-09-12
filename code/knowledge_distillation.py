import os
import hydra

import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from transformers import (
    AutoConfig,
    AutoModel,
    BertConfig,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tensorboardX import SummaryWriter

from data.datasets import load_shinra_results_dataset
from data.file_utils import FileUtils
from model import BertForKnowledgeDistillationIOB2Tagging
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
            dataset.labeled_class_counts,
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
    early_stopping_count = 0
    for epoch in range(1,cfg.train.epoch+1):
        model.train()
        train_ens_outputs = []
        for batch in tqdm.tqdm(dataloader, desc=f"TRAIN {epoch}"):
            steps += 1

            logits, ens_logits = model(
                input_ids=batch["input_ids"].to(cfg.device.device),
                attention_mask=batch["attention_mask"].to(cfg.device.device),
            )
            labels=batch["labels"].to(cfg.device.device)
            labels = labels.view(-1, len(dataset.systems), len(dataset.attributes))
            labels = labels.transpose(0, 1)

            losses = []
            for system, (t_labels, t_logits) in enumerate(zip(labels, logits.permute(1, 0, 2, 3, 4))):
                t_labels = t_labels.contiguous().view(-1)
                act_t_logits = t_logits.contiguous().view(-1, 3)[t_labels >= 0]
                act_t_labels = t_labels[t_labels >= 0]

                loss = loss_fct(act_t_logits, act_t_labels)
                losses.append(loss)

            losses = torch.stack(losses)

            gold_labels=batch["gold_labels"].view(-1).to(cfg.device.device)
            act_ens_logits = ens_logits.view(-1, 3)[gold_labels>=0]
            act_gold_labels = gold_labels[gold_labels>=0]

            ens_loss = loss_fct(act_ens_logits, act_gold_labels)

            writer.add_scalars("Train_Each_Loss", dict(zip(dataset.systems, losses)), steps)
            writer.add_scalar("Train_Ens_Loss", ens_loss.item(), steps)

            loss = losses.mean()

            if not torch.isnan(ens_loss).item():
                loss += ens_loss / dataset.labeled_rate
                loss /= 2

            ens_outputs = ens_logits.argmax(dim=-1)
            train_ens_outputs += [*zip(
                batch["pageid"],
                batch["part_id"].tolist(),
                batch["offsets"].tolist(),
                ens_outputs.cpu().tolist()
            )]

            optimizer.zero_grad()
            if cfg.device.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            writer.add_scalar("Train__Loss", loss.item(), steps)

        ens_score = dataset.labeled_dataset.evaluate(train_ens_outputs)
        writer.add_scalars("Score__Train___Ens_MicroAve", ens_score["micro_ave"], epoch)

        dev_score, dev_ens_score = eval(cfg, dev_dataset, model)
        writer.add_scalars("Score__Dev___Ens_MicroAve", dev_ens_score["micro_ave"], epoch)

        print(dev_ens_score["micro_ave"])
        if best_score is None or dev_ens_score["micro_ave"]["F1"] >= best_score["F1"]:
            best_score = dev_ens_score["micro_ave"]
            if cfg.model.output_dir is not None:
                output_dir = hydra.utils.to_absolute_path(cfg.model.output_dir)
                print(f"Save model to {output_dir}")
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save(
                    output_dir,
                    attributes=dataset.attributes,
                    systems=dataset.systems,
                    class_weight_scheme=cfg.train.class_weight_scheme,
                    weighting_method=cfg.distillation.method
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
    eval_ens_outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="EVAL"):
            outputs, ens_outputs = model(
                input_ids=batch["input_ids"].to(cfg.device.device),
                attention_mask=batch["attention_mask"].to(cfg.device.device)
            )
            ens_outputs = ens_outputs.argmax(dim=-1)
            eval_ens_outputs += [*zip(
                batch["pageid"],
                batch["part_id"].tolist(),
                batch["offsets"].tolist(),
                ens_outputs.cpu().tolist()
            )]

    model.train()
    return None, dataset.labeled_dataset.evaluate(eval_ens_outputs)

@hydra.main(config_path="../config", config_name="knowledge_distillation")
def main(cfg):
    cfg.device.n_gpu = torch.cuda.device_count()
    cfg.device.device = "cuda" if torch.cuda.is_available() and not cfg.device.no_cuda else "cpu"
    set_seed(cfg.seed)
    train_dataset, dev_dataset = load_shinra_results_dataset(cfg)

    model_dir = hydra.utils.to_absolute_path(cfg.model.file_dir)
    config = AutoConfig.from_pretrained(
        os.path.join(model_dir, cfg.model.config_name),
        num_labels=len(train_dataset.attributes),
        add_pooling_layer=False
    )
    config.num_systems = len(train_dataset.systems)
    bert = AutoModel.from_pretrained(
        os.path.join(model_dir, cfg.model.model_name),
        config=config
    )
    model = BertForKnowledgeDistillationIOB2Tagging(
        config,
        bert,
        class_counts=train_dataset.class_counts,
        class_weight_scheme=cfg.train.class_weight_scheme,
        focal_loss_gamma=cfg.train.focal_loss_gamma,
        balanced_loss_beta=cfg.train.balanced_loss_beta
    )

    model.to(cfg.device.device)

    # TensorboardX
    log_dir = hydra.utils.to_absolute_path(cfg.log_dir)
    writer = SummaryWriter(log_dir)

    train(cfg, train_dataset, dev_dataset, model, writer)

if __name__ == '__main__':
    main()
