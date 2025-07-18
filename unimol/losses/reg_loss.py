# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from sklearn.metrics import r2_score

@register_loss("finetune_mse")
class FinetuneMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        if sample["target"]["finetune_target"].size(1) != reg_output.size(1):
            sample["target"]["finetune_target"] = sample["target"]["finetune_target"][:, 0:1] # Hack for wrong label size
        
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task, add aggregate acc and loss score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                        "smi": smi_list,
                    }
                )
                mae = np.abs(df["predict"] - df["target"]).mean()
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                df = df.groupby("smi").mean()
                agg_mae = np.abs(df["predict"] - df["target"]).mean()
                agg_mse = ((df["predict"] - df["target"]) ** 2).mean()

                metrics.log_scalar(f"{split}_mae", mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_mse", agg_mse, sample_size, round=3)
                metrics.log_scalar(
                    f"{split}_agg_rmse", np.sqrt(agg_mse), sample_size, round=4
                )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_mae")
class FinetuneMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )
        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        loss = F.l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss


@register_loss("finetune_smooth_mae")
class FinetuneSmoothMAELoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )

        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        
        
        loss = F.smooth_l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            # # agg_mae = np.abs(y_pred - y_true).mean(axis=0).mean(axis=1)
            agg_mae = np.abs(y_pred - y_true)
            metrics.log_scalar(f"{split}_agg_mae", np.mean(agg_mae), sample_size, round=4)
            if num_task > 1:
                for i in range(num_task):
                    metrics.log_scalar(f"{split}_agg_mae_{i}", np.mean(agg_mae[:, i]), sample_size, round=4)

            agg_r2 = []
            for i in range(num_task):
                agg_r2.append(r2_score(y_true[:, i], y_pred[:, i]))
            # agg_r2 /= num_task
            metrics.log_scalar(f"{split}_agg_r2", np.mean(np.array(agg_r2)), sample_size, round=4)
            if num_task > 1:
                for i in range(num_task):
                    metrics.log_scalar(f"{split}_agg_r2_{i}", agg_r2[i], sample_size, round=4)

@register_loss("finetune_smooth_mae_ranking")
class FinetuneSmoothMAERankingLoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        net_input1 = {
                    "src_tokens": sample['net_input']["src_tokens1"],
                    "src_coord": sample['net_input']["src_coord1"],
                    "src_distance": sample['net_input']["src_distance1"],
                    "src_edge_type": sample['net_input']["src_edge_type1"],
                }
        
        net_input2 = {
                    "src_tokens": sample['net_input']["src_tokens1"],
                    "src_coord": sample['net_input']["src_coord1"],
                    "src_distance": sample['net_input']["src_distance1"],
                    "src_edge_type": sample['net_input']["src_edge_type1"],
                }
        
        net_output1 = model(
            **net_input1,
            masked_tokens=None,
            features_only=True,
            classification_head_name=None,
        )

        net_output2 = model(
            **net_input2,
            masked_tokens=None,
            features_only=True,
            classification_head_name=None,
        )

        mol1_rep = net_output1[0]
        mol2_rep = net_output2[0]


        # Check idx of classification and regression
        rank_idxs = sample["target"]["rank_target"] != 123456789
        reg_idxs = sample["target"]["reg_target"] != 123456789

        loss = 0
        reg_loss = torch.zeros(1).to(mol1_rep.device)
        rank_loss = torch.zeros(1).to(mol1_rep.device)

        if reg_idxs.sum() > 0:
            reg_output = model.classification_heads['rank-regressor'](mol1_rep, None)
            predicts = reg_output.float().squeeze(-1)
            targets = sample["target"]["reg_target"].float()

            predicts = predicts[reg_idxs]
            targets = targets[reg_idxs]
           
            reg_loss = F.smooth_l1_loss(
                predicts,
                targets,
                reduction="sum" if reduce else "none",
            )

            loss += reg_loss

        if rank_idxs.sum() > 0:
            logit_output = model.classification_heads['rank-regressor'](mol1_rep, mol2_rep).squeeze(-1)   
            logit_output = logit_output.squeeze(-1)   

            pred = logit_output.float()
            targets = sample["target"]["rank_target"].float()

            pred = pred[rank_idxs]
            targets = targets[rank_idxs]

            rank_loss = F.binary_cross_entropy_with_logits(
                pred,
                targets,
                reduction="sum" if reduce else "none",
            )

            loss += 0.1 * rank_loss

        sample_size = sample["target"]["rank_target"].size(0)
        if not self.training:
            # probs = torch.sigmoid(logit_output.float()).view(-1, logit_output.size(-1))
            logging_output = {
                "loss": loss.data,
                # "rank_loss": rank_loss.data,
                "reg_loss": reg_loss.data,
                # "prob": probs.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["reg_target"].view(-1).data,
                "num_task": self.args.num_classes,
                "sample_size": sample_size,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["reg_target"].size(0),
                "smi_name": sample["smi_name1"] # Assuming smi_name1 is for regression
            }
        else:
            logging_output = {
                "loss": loss.data,
                "rank_loss": rank_loss.data,
                "reg_loss": reg_loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["rank_target"].size(0),
            }
        return loss, sample_size, logging_output
    

    def compute_loss(self, model, net_output, sample, reduce=True):
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = (
            sample["target"]["finetune_target"].view(-1, self.args.num_classes).float()
        )

        if self.task.mean and self.task.std:
            targets_mean = torch.tensor(self.task.mean, device=targets.device)
            targets_std = torch.tensor(self.task.std, device=targets.device)
            targets = (targets - targets_mean) / targets_std
        
        
        loss = F.smooth_l1_loss(
            predicts,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "rank_loss", sum(log.get("rank_loss", 0) for log in logging_outputs) / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "reg_loss", sum(log.get("reg_loss", 0) for log in logging_outputs) / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("predict") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            # # agg_mae = np.abs(y_pred - y_true).mean(axis=0).mean(axis=1)
            agg_mae = np.abs(y_pred - y_true).mean()
            metrics.log_scalar(f"{split}_agg_mae", agg_mae, sample_size, round=4)

            agg_r2 = 0
            for i in range(num_task):
                agg_r2 += r2_score(y_true[:, i], y_pred[:, i])
            agg_r2 /= num_task
            metrics.log_scalar(f"{split}_agg_r2", agg_r2, sample_size, round=4)

@register_loss("finetune_mse_pocket")
class FinetuneMSEPocketLoss(FinetuneMSELoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        reg_output = net_output[0]
        loss = self.compute_loss(model, reg_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            if self.task.mean and self.task.std:
                targets_mean = torch.tensor(self.task.mean, device=reg_output.device)
                targets_std = torch.tensor(self.task.std, device=reg_output.device)
                reg_output = reg_output * targets_std + targets_mean
            logging_output = {
                "loss": loss.data,
                "predict": reg_output.view(-1, self.args.num_classes).data,
                "target": sample["target"]["finetune_target"]
                .view(-1, self.args.num_classes)
                .data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
            if predicts.size(-1) == 1:
                # single label regression task
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                df = pd.DataFrame(
                    {
                        "predict": predicts.view(-1).cpu(),
                        "target": targets.view(-1).cpu(),
                    }
                )
                mse = ((df["predict"] - df["target"]) ** 2).mean()
                metrics.log_scalar(f"{split}_mse", mse, sample_size, round=3)
                metrics.log_scalar(f"{split}_rmse", np.sqrt(mse), sample_size, round=4)
