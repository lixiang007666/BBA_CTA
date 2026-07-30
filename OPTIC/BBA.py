import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger
from utils.convert import AdaBN
from utils.metrics import calculate_metrics
from networks.ResUnet_TTA import ResUnet
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
from tqdm import tqdm
import torch.nn.functional as F


torch.set_num_threads(1)


@torch.no_grad()
def dual_reliability_targets(logits, features, gamma=0.75, eta=0.05, eps=1e-6):
    """Implement the entropy and prototype filters in Eqs. (7)-(12)."""
    probabilities = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pseudo_labels = (probabilities >= gamma).to(probabilities.dtype)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
    entropy_mask = entropy < eta

    if features.shape[-2:] != probabilities.shape[-2:]:
        features = F.interpolate(
            features, size=probabilities.shape[-2:], mode="bilinear", align_corners=False
        )

    feature_grid = features.unsqueeze(1)
    object_weights = entropy_mask * pseudo_labels * probabilities
    background_weights = entropy_mask * (1.0 - pseudo_labels) * (1.0 - probabilities)

    object_count = object_weights.sum(dim=(-2, -1), keepdim=True)
    background_count = background_weights.sum(dim=(-2, -1), keepdim=True)
    object_prototype = (
        feature_grid * object_weights.unsqueeze(2)
    ).sum(dim=(-2, -1), keepdim=True) / object_count.unsqueeze(2).clamp_min(eps)
    background_prototype = (
        feature_grid * background_weights.unsqueeze(2)
    ).sum(dim=(-2, -1), keepdim=True) / background_count.unsqueeze(2).clamp_min(eps)

    object_distance = ((feature_grid - object_prototype) ** 2).sum(dim=2)
    background_distance = ((feature_grid - background_prototype) ** 2).sum(dim=2)
    prototype_mask = torch.where(
        pseudo_labels.bool(),
        object_distance < background_distance,
        background_distance < object_distance,
    )
    valid_prototypes = (object_count > eps) & (background_count > eps)
    reliability_mask = entropy_mask & prototype_mask & valid_prototypes
    return pseudo_labels, reliability_mask.to(probabilities.dtype)


def reliable_bce_loss(logits, pseudo_labels, reliability_mask):
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, pseudo_labels, reduction="none"
    )
    return (per_pixel * reliability_mask).sum() / reliability_mask.sum().clamp_min(1.0)


class BBA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, "BBA")
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + ".log")
        sys.stdout = Logger(log_path, sys.stdout)
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.eta = config.eta

        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            if target != "REFUGE_Valid":
                target_test_csv.append(target + "_train.csv")
                target_test_csv.append(target + "_test.csv")
            else:
                target_test_csv.append(target + ".csv")
        ts_img_list, ts_label_list = convert_labeled_list(
            config.dataset_root, target_test_csv
        )
        if config.max_samples is not None:
            ts_img_list = ts_img_list[: config.max_samples]
            ts_label_list = ts_label_list[: config.max_samples]
        target_test_dataset = OPTIC_dataset(
            config.dataset_root,
            ts_img_list,
            ts_label_list,
            config.image_size,
            img_normalize=True,
        )
        self.target_test_loader = DataLoader(
            dataset=target_test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_wo_transform,
            num_workers=config.num_workers,
        )
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(
            config.model_root, str(config.Source_Dataset)
        )  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device

        self.bn_tau = config.bn_tau

        self.iters = config.iters
        self.rounds = config.rounds

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print("***" * 20)

    def build_model(self):
        self.model = ResUnet(
            resnet=self.backbone,
            num_classes=self.out_ch,
            pretrained=False,
            newBN=AdaBN,
            bn_tau=self.bn_tau,
        ).to(self.device)

        checkpoint = torch.load(
            os.path.join(self.load_model, "last-Res_Unet.pth"), weights_only=True
        )

        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in checkpoint.items()
            if k in model_dict and "my_module" not in k
        }
        model_dict.update(pretrained_dict)

        self.model.load_state_dict(model_dict, strict=False)

        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        print(f"Trainable PEOA parameters: {sum(p.numel() for p in trainable_params):,}")

        if self.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "Adam":
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        metric_dict = ["Disc_Dice", "Disc_ASD", "Cup_Dice", "Cup_ASD"]

        result_save_dir = "./prediction_results/"
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)

        for pass_num in range(self.rounds):
            metrics_test = [[], [], [], []]
            reliable_pixels = 0
            total_pixels = 0
            for batch, data in enumerate(
                tqdm(self.target_test_loader, desc="Processing batches", ncols=100)
            ):
                x, y = data["data"], data["mask"]
                x = torch.from_numpy(x).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.long)
                x, y = x.to(self.device), y.to(self.device)

                self.model.eval()
                self.model.change_BN_status(new_sample=True)
                with torch.no_grad():
                    initial_logits, _, initial_features = self.model(x)
                pseudo_label, reliability_mask = dual_reliability_targets(
                    initial_logits, initial_features, gamma=self.gamma, eta=self.eta
                )
                reliable_pixels += int(reliability_mask.sum().item())
                total_pixels += reliability_mask.numel()
                self.model.change_BN_status(new_sample=False)

                for tr_iter in range(self.iters):
                    pred_logit, fea, head_input = self.model(x)
                    times, bn_loss = 0, 0
                    for nm, m in self.model.named_modules():
                        if isinstance(m, AdaBN):
                            bn_loss += m.bn_loss
                            times += 1
                    bn_loss = bn_loss / max(times, 1)
                    seg_loss = reliable_bce_loss(
                        pred_logit, pseudo_label, reliability_mask
                    )
                    loss = seg_loss + self.alpha * bn_loss

                    if batch < 3:
                        print(
                            f"batch={batch} iter={tr_iter + 1} "
                            f"reliable={reliability_mask.mean().item():.6f} "
                            f"seg_loss={seg_loss.item():.6f} "
                            f"bn_loss={bn_loss.item():.6f} total={loss.item():.6f}"
                        )

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.model.change_BN_status(new_sample=False)

                # Inference
                self.model.eval()
                # self.prompt.eval()
                with torch.no_grad():
                    pred_logit, fea, head_input = self.model(x)
                self.model.commit_memory()

                # Calculate the metrics
                seg_output = torch.sigmoid(pred_logit)
                metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
                for i in range(len(metrics)):
                    assert isinstance(
                        metrics[i], list
                    ), "The metrics value is not list type."
                    metrics_test[i] += metrics[i]

            test_metrics_y = np.mean(metrics_test, axis=1)
            print_test_metric_mean = {}
            for i in range(len(test_metrics_y)):
                print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
            print(f"Round {pass_num + 1} Test Metrics: ", print_test_metric_mean)
            print(
                "Reliable pseudo-label pixels:",
                f"{100.0 * reliable_pixels / max(total_pixels, 1):.4f}%",
            )
            print(
                "Mean Dice:",
                (
                    print_test_metric_mean["Disc_Dice"]
                    + print_test_metric_mean["Cup_Dice"]
                )
                / 2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument(
        "--Source_Dataset",
        type=str,
        default="RIM_ONE_r3",
        help="RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS",
    )
    parser.add_argument(
        "--target_datasets",
        nargs="+",
        choices=["RIM_ONE_r3", "REFUGE", "ORIGA", "REFUGE_Valid", "Drishti_GS"],
        help="Ordered target-domain stream; defaults to every non-source domain.",
    )

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)

    # Model
    parser.add_argument(
        "--backbone", type=str, default="resnet34", help="resnet34/resnet50"
    )
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=2)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="Adam", help="SGD/Adam")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.99)  # momentum in SGD
    parser.add_argument("--beta1", type=float, default=0.9)  # beta1 in Adam
    parser.add_argument("--beta2", type=float, default=0.99)  # beta2 in Adam
    parser.add_argument("--weight_decay", type=float, default=0.00)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--bn_tau", type=float, default=0.01)

    # Path
    parser.add_argument("--path_save_log", type=str, default="./logs")
    parser.add_argument("--model_root", type=str, default="./models")
    parser.add_argument("--dataset_root", type=str, default="./Fundus")

    # Cuda (default: the first available device)
    parser.add_argument("--device", type=str, default="cuda:0")

    config = parser.parse_args()

    all_datasets = ["RIM_ONE_r3", "REFUGE", "ORIGA", "REFUGE_Valid", "Drishti_GS"]
    config.Target_Dataset = config.target_datasets or [
        dataset for dataset in all_datasets if dataset != config.Source_Dataset
    ]
    if config.Source_Dataset in config.Target_Dataset:
        parser.error("The source dataset cannot also be a target dataset.")

    TTA = BBA(config)
    # pdb.set_trace()
    # print(TTA.model)
    TTA.run()
