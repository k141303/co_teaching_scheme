import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tqdm
import hydra

import torch
from torch.utils.data import DataLoader

from data.file_utils import FileUtils
from data.datasets import load_target_dataset
from model import BertForDistillingIOB2Tagging, BertForIOB2Tagging, BertForKnowledgeDistillationIOB2Tagging

def predict(cfg, dataset, model):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=cfg.pred.batch_size, num_workers = cfg.data.num_workers)

    eval_outputs = []
    ensemble_outputs = []
    mean_outputs = []
    system_mean_outputs = []
    raw_outputs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="EVAL"):
            outputs, ens_outputs = model(
                input_ids=batch["input_ids"].to(cfg.device.device),
                attention_mask=batch["attention_mask"].to(cfg.device.device)
            )

            if outputs is not None and ens_outputs is not None:
                m_outputs = [ens_outputs.softmax(dim=-1), outputs.softmax(dim=-1).mean(dim=1)]
                m_outputs = torch.stack(m_outputs).mean(dim=0).argmax(dim=-1)

                mean_outputs += [*zip(
                    batch["pageid"],
                    batch["part_id"].tolist(),
                    batch["offsets"].tolist(),
                    m_outputs.cpu().tolist()
                )]
                system_mean_outputs += [*zip(
                    batch["pageid"],
                    batch["part_id"].tolist(),
                    batch["offsets"].tolist(),
                    outputs.softmax(dim=-1).mean(dim=1).argmax(dim=-1).cpu().tolist()
                )]

            if ens_outputs is not None:
                ens_outputs = ens_outputs.argmax(dim=-1).cpu().tolist()

                ensemble_outputs += [*zip(
                    batch["pageid"],
                    batch["part_id"].tolist(),
                    batch["offsets"].tolist(),
                    ens_outputs
                )]
                raw_outputs += [{"page_id":pageid, "part_id":part_id, "iob2tag":output} for pageid, part_id, output in zip(
                    batch["pageid"],
                    batch["part_id"].tolist(),
                    ens_outputs
                )]

            if outputs is not None:
                outputs = outputs.argmax(dim=-1).cpu().tolist()
                eval_outputs += [*zip(
                    batch["pageid"],
                    batch["part_id"].tolist(),
                    batch["offsets"].tolist(),
                    outputs
                )]

    if len(eval_outputs) != 0:
        eval_outputs = dataset.convert(eval_outputs)

    if len(ensemble_outputs) != 0:
        ensemble_outputs = dataset.convert_single(ensemble_outputs)

    if len(mean_outputs) != 0:
        mean_outputs = dataset.convert_single(mean_outputs)

    if len(system_mean_outputs) != 0:
        system_mean_outputs = dataset.convert_single(system_mean_outputs)

    return eval_outputs, ensemble_outputs, mean_outputs, system_mean_outputs, raw_outputs

@hydra.main(config_path="../config", config_name="prediction_config")
def main(cfg):
    cfg.device.n_gpu = torch.cuda.device_count()
    cfg.device.device = "cuda" if torch.cuda.is_available() and not cfg.device.no_cuda else "cpu"

    model_dir = hydra.utils.to_absolute_path(cfg.model.file_dir)
    config, _, model = eval(cfg.model.model_class).load(
        model_dir,
        config_name=cfg.model.config_name,
        model_name=cfg.model.model_name
    )
    target_dataset = load_target_dataset(cfg, config.get("systems"), config["attributes"])

    model.to(cfg.device.device)

    if cfg.device.fp16 and not cfg.device.no_cuda:
        try:
            import apex
            from apex import amp
        except ModuleNotFoundError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = apex.amp.initialize(model, opt_level=cfg.device.fp16_opt_level)

    if cfg.device.n_gpu != 1:
        model = torch.nn.DataParallel(model)

    submit_formats, ensemble_submit_format, mean_submit_format, system_mean_submit_format, raw_outputs = predict(cfg, target_dataset, model)

    output_dir = os.path.join(model_dir, "predictions") if cfg.data.save_dir is None else hydra.utils.to_absolute_path(cfg.data.save_dir)
    for system in submit_formats:
        output_path = os.path.join(output_dir, system, "results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        FileUtils.JsonL.save(
            output_path,
            submit_formats[system]
        )
    output_path = os.path.join(output_dir, "ensemble", "results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    FileUtils.JsonL.save(
        output_path,
        ensemble_submit_format
    )
    output_path = os.path.join(output_dir, "mean", "results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    FileUtils.JsonL.save(
        output_path,
        mean_submit_format
    )
    output_path = os.path.join(output_dir, "system_mean", "results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    FileUtils.JsonL.save(
        output_path,
        system_mean_submit_format
    )
    if cfg.data.save_raw_outouts:
        output_path = os.path.join(output_dir, "raw", "results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        FileUtils.JsonL.save(
            output_path,
            raw_outputs
        )


if __name__ == '__main__':
    main()
