import os
import torch
from torch import nn

import math
from collections import defaultdict

from model.loss_fct import ClassBalancedFocalCELoss

from transformers import BertConfig, AutoModel, RobertaConfig
from transformers.modeling_bert import BertEncoder

from data.file_utils import FileUtils
from model.modeling_utils import MyBertEncoder, PrivateModelForIOB2Tagging
from model.loss_fct import ClassBalancedFocalCELoss

class BertForIOB2Tagging(nn.Module):
    def __init__(
            self,
            config,
            bert,
            class_counts=None,
            class_weight_scheme="no_weight",
            focal_loss_gamma=2,
            balanced_loss_beta=0.9999,
            freeze_share_model=False
        ):
        super().__init__()
        self.bert = bert
        self.config = config
        self.num_labels = config.num_labels
        self.freeze_share_model = freeze_share_model
        self.gelu = torch.nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer = PrivateModelForIOB2Tagging(config, self.num_labels)


        self.class_weight_scheme = class_weight_scheme
        if self.class_weight_scheme == "no_weight":
            self.class_weights = torch.tensor([1.0, 1.0, 1.0])
        elif self.class_weight_scheme == "uniform_weight":
            self.class_weights = torch.tensor([1.0, 100.0, 100.0])
        elif self.class_weight_scheme == "balanced_focal":
            self.focal_loss_gamma = focal_loss_gamma
            self.class_counts = torch.tensor([class_counts[i] for i in range(3)])
            self.balanced_loss_beta = balanced_loss_beta
            self.class_counts = self.class_counts.tolist()
        else:
             raise NotImplementedError()

    def forward(
        self,
        **kwargs
    ):
        if self.freeze_share_model:
            with torch.no_grad():
                hidden_state, *_  = self.bert(**kwargs)
        else:
            hidden_state, *_  = self.bert(**kwargs)

        hidden_state = self.gelu(hidden_state)

        hidden_state = self.dropout(hidden_state)
        logits = self.layer(hidden_state)
        bsz, seq_len, _ = logits.size()
        logits = logits.view(bsz, seq_len, self.num_labels, 3)

        return None, logits

    def save(self, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.state_dict(),
            os.path.join(output_dir, "pytorch_model.bin")
        )
        kwargs.update({
            "bert_config": self.config.to_dict(),
        })
        FileUtils.Json.save(
            os.path.join(output_dir, "config.json"),
            kwargs
        )

    @classmethod
    def load(cls, model_dir, config_name="config.json", model_name="pytorch_model.bin"):
        config_path = os.path.join(model_dir, config_name)
        config = FileUtils.Json.load(config_path)
        bert_config = RobertaConfig.from_dict(config["bert_config"])

        bert = AutoModel.from_config(bert_config)
        model = cls(bert_config, bert)

        state_dict = torch.load(
            os.path.join(model_dir, model_name)
        )
        model.load_state_dict(state_dict)
        return config, bert_config, model

class BertForDistillingIOB2Tagging(nn.Module):
    def __init__(self, config, bert, class_counts=None, class_weight_scheme="no_weight", focal_loss_gamma=2, balanced_loss_beta=0.999999):
        super().__init__()
        self.bert = bert
        self.config = config

        self.class_weight_scheme = class_weight_scheme
        if self.class_weight_scheme == "no_weight":
            self.class_weights = torch.tensor([1.0, 1.0, 1.0])
        elif self.class_weight_scheme == "uniform_weight":
            self.class_weights = torch.tensor([1.0, 500.0, 500.0])
        elif self.class_weight_scheme == "balanced_focal":
            self.focal_loss_gamma = focal_loss_gamma
            self.balanced_loss_beta = balanced_loss_beta
            self.class_counts = torch.tensor([[class_counts[s][i] for i in range(3)] for s in class_counts])
            self.class_counts = torch.sum(self.class_counts, dim=0)
            self.class_counts = self.class_counts.tolist()
        else:
             raise NotImplementedError()

        self.gelu = torch.nn.GELU()
        self.num_labels = config.num_labels
        self.num_systems = config.num_systems
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.private_models = nn.ModuleList(
            [PrivateModelForIOB2Tagging(config, self.num_labels) for i in range(self.num_systems)]
        )

    def forward(
        self,
        labels = None,
        return_hidden_state = False,
        hidden_state = None,
        **kwargs
    ):
        if hidden_state is None:
            hidden_state, _ = self.bert(**kwargs)
            hidden_state = self.gelu(hidden_state)

        if return_hidden_state:
            return hidden_state

        hidden_state = self.dropout(hidden_state)
        logits = [private_model(hidden_state) for private_model in self.private_models]
        logits = torch.stack(logits)

        logits = logits.transpose(0, 1) #(num_systems, batch_size, seq_len, num_labels*3)=> (batch_size, num_systems, seq_len, num_labels*3)
        bsz, _, seq_len, _ = logits.size()
        logits = logits.contiguous().view(bsz, self.num_systems, seq_len, self.num_labels, 3)

        if labels is not None:
            labels = labels.view(-1, self.num_systems, self.num_labels)
            labels = labels.transpose(0, 1)

            losses = []

            if self.class_weight_scheme == "balanced_focal":
                loss_fct = ClassBalancedFocalCELoss(self.class_counts, 3, beta=self.balanced_loss_beta, gamma=self.focal_loss_gamma, size_average=False)
            elif self.class_weight_scheme in ["no_weight", "uniform_weight"]:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(labels.device), size_average=False)
            else:
                raise NotImplementedError()

            for system, (t_labels, t_logits) in enumerate(zip(labels, logits.permute(1, 0, 2, 3, 4))):
                t_labels = t_labels.contiguous().view(-1)
                act_t_logits = t_logits.contiguous().view(-1, 3)[t_labels >= 0]
                act_t_labels = t_labels[t_labels >= 0]

                loss = loss_fct(act_t_logits, act_t_labels)
                losses.append(loss)

            return losses, logits

        return logits, None

    @classmethod
    def load(cls, model_dir, config_name="config.json", model_name="pytorch_model.bin"):
        config_path = os.path.join(model_dir, config_name)
        config = FileUtils.Json.load(config_path)
        bert_config = RobertaConfig.from_dict(config["bert_config"])

        bert = AutoModel.from_config(bert_config)
        model = cls(bert_config, bert)

        state_dict = torch.load(
            os.path.join(model_dir, model_name)
        )
        model.load_state_dict(state_dict)
        return config, bert_config, model

    def save(self, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            self.state_dict(),
            os.path.join(output_dir, "pytorch_model.bin")
        )
        kwargs.update({
            "bert_config": self.config.to_dict()
        })
        FileUtils.Json.save(
            os.path.join(output_dir, "config.json"),
            kwargs
        )

class BertForKnowledgeDistillationIOB2Tagging(BertForDistillingIOB2Tagging):
    def __init__(self, config, bert, class_counts=None, class_weight_scheme="no_weight", focal_loss_gamma=2, balanced_loss_beta=0.999999):
        super().__init__(config, bert, class_counts, class_weight_scheme, focal_loss_gamma, balanced_loss_beta)
        self.layer = PrivateModelForIOB2Tagging(config, self.num_labels)

    def forward(
        self,
        return_hidden_state = False,
        hidden_state = None,
        **kwargs
    ):
        if hidden_state is None:
            hidden_state, _ = self.bert(**kwargs)
            hidden_state = self.gelu(hidden_state)

        if return_hidden_state:
            return hidden_state

        hidden_state = self.dropout(hidden_state)
        logits = [private_model(hidden_state) for private_model in self.private_models]
        logits = torch.stack(logits)

        logits = logits.transpose(0, 1) #(num_systems, batch_size, seq_len, num_labels*3)=> (batch_size, num_systems, seq_len, num_labels*3)
        bsz, _, seq_len, _ = logits.size()
        logits = logits.contiguous().view(bsz, self.num_systems, seq_len, self.num_labels, 3)

        ens_logits = self.layer(hidden_state)
        ens_logits = ens_logits.view(bsz, seq_len, -1, 3)

        return logits, ens_logits
