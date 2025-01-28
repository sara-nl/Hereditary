"""FETS-swin-unet: A Flower / PyTorch app."""

from functools import partial

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

from fets_swin_unet.task import get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, net, trainloader, valloader, local_epochs, total_epochs, batch_size, roi, sw_batch_size, infer_overlap
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        self.dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.roi = roi
        self.sw_batch_size = sw_batch_size
        self.infer_overlap = infer_overlap

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_epochs)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.optimizer,
            self.local_epochs,
            self.dice_loss,
            self.batch_size,
            self.device,
        )
        self.scheduler.step()
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        roi = self.roi
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=[roi[0], roi[1], roi[2]],
            sw_batch_size=self.sw_batch_size,
            predictor=self.net,
            overlap=self.infer_overlap,
        )
        loss, class_accuracies = test(
            self.net, self.valloader, self.dice_acc, self.device, self.model_inferer, self.post_sigmoid, self.post_pred
        )
        return (
            loss,
            len(self.valloader.dataset),
            {
                "accuracy": loss,  # Overall mean dice score
                "dice_tc": class_accuracies[0],  # Tumor Core
                "dice_wt": class_accuracies[1],  # Whole Tumor
                "dice_et": class_accuracies[2],  # Enhancing Tumor
            },
        )


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    local_epochs = context.run_config["local-epochs"]
    num_rounds = context.run_config["num-server-rounds"]
    total_epochs = num_rounds * local_epochs
    batch_size = context.run_config["batch-size"]
    infer_overlap = context.run_config["infer-overlap"]
    sw_batch_size = context.run_config["sw-batch-size"]
    roi = [int(i) for i in context.run_config["roi"].split(",")]
    trainloader, valloader = load_data(batch_size, partition_id, roi)
    net = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )
    # Return Client instance
    return FlowerClient(
        net, trainloader, valloader, local_epochs, total_epochs, batch_size, roi, sw_batch_size, infer_overlap
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
