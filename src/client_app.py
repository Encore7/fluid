import json
import os

import torch
import torch.nn as nn
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import DataLoader
from src.ml_models.cnn import Net
from src.ml_models.utils import (
    get_weights,
    initialize_model,
    rapid_train,
    set_weights,
    test,
    train,
)
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_number,
        batch_size,
        local_epochs,
        learning_rate,
        dataset_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        drift_start_round,
        drift_end_round,
        drift_clients,
        abrupt_drift_labels_swap,
        incremental_drift_rounds,
        mode,
        num_of_batches,
        num_of_clients,
    ):
        super().__init__()
        if mode == "adaptive-fluid":
            self.net_rrt = initialize_model(Net())
            self.net_fedau = initialize_model(Net())

        self.net = initialize_model(Net())

        self.client_number = client_number
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dataset_folder_path = dataset_folder_path
        self.client_data_folder_path = os.path.join(
            dataset_folder_path, f"client_{client_number}"
        )
        self.dataset_input_feature = dataset_input_feature
        self.dataset_target_feature = dataset_target_feature
        self.drift_start_round = drift_start_round
        self.drift_end_round = drift_end_round
        self.drift_clients = drift_clients
        self.abrupt_drift_labels_swap = abrupt_drift_labels_swap
        self.incremental_drift_rounds = incremental_drift_rounds
        self.mode = mode
        self.num_of_batches = num_of_batches
        self.num_of_clients = num_of_clients

        self.device = self._get_device()

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)
        self.logger.info("Client %s initiated", self.client_number)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _get_drift_config(self, current_round, adaptive_fluid_model=None):
        is_drifted = False
        percentage_to_swap = 1.0
        create_fedau_dataloader = False
        invert_drift_selection = False

        if (
            self.drift_start_round <= current_round < self.drift_end_round
            and len(self.drift_clients) > self.num_of_clients / 2
        ):
            invert_drift_selection = True
            self.drift_clients = [
                i for i in range(self.num_of_clients) if i not in self.drift_clients
            ]

        if (
            (self.drift_start_round <= current_round < self.drift_end_round)
            and self.mode != "rapid-retraining"
            and adaptive_fluid_model != "rrt-only"
        ):
            if self.client_number in self.drift_clients:
                is_drifted = True

                keys = sorted(map(int, self.incremental_drift_rounds.keys()))
                for i in range(len(keys) - 1):
                    if keys[i] <= current_round < keys[i + 1]:
                        percentage_to_swap = self.incremental_drift_rounds[str(keys[i])]
                if current_round >= keys[-1]:  # Handle the last range
                    percentage_to_swap = self.incremental_drift_rounds[str(keys[-1])]

                if invert_drift_selection:
                    percentage_to_swap = 1.0 - percentage_to_swap

                if (
                    self.mode == "fedau"
                    or self.mode == "fluid"
                    or self.mode == "adaptive-fluid"
                ):
                    create_fedau_dataloader = True
            elif invert_drift_selection and (
                self.mode == "fedau"
                or self.mode == "fluid"
                or self.mode == "adaptive-fluid"
            ):
                create_fedau_dataloader = True

        return (
            is_drifted,
            percentage_to_swap,
            create_fedau_dataloader,
            invert_drift_selection,
        )

    def abrupt_drift_labels_swap_modify(self, current_round):
        if current_round >= 35 and current_round < 65:
            self.abrupt_drift_labels_swap = [{"label1": 1, "label2": 2}]
        elif current_round >= 65:
            self.abrupt_drift_labels_swap = [
                {"label1": 1, "label2": 2},
                {"label1": 5, "label2": 7},
            ]

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round")
        adaptive_fluid_model = config.get("adaptive_fluid_model", None)

        self.abrupt_drift_labels_swap_modify(current_round)

        self.logger.info("current_round %s", current_round)

        (
            is_drifted,
            percentage_to_swap,
            create_fedau_dataloader,
            invert_drift_selection,
        ) = self._get_drift_config(current_round, adaptive_fluid_model)

        self.logger.info(
            "is_drifted %s, percentage_to_swap %s, create_fedau_dataloader %s, invert_drift_selection %s",
            is_drifted,
            percentage_to_swap,
            create_fedau_dataloader,
            invert_drift_selection,
        )

        # Create data loader instance for transforms
        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
            dataset_target_feature=self.dataset_target_feature,
        )

        dataloader_props = {
            "client_folder_path": self.client_data_folder_path,
            "batch_size": self.batch_size,
            "num_of_batches": self.num_of_batches,
            "dataset_input_feature": self.dataset_input_feature,
            "dataset_target_feature": self.dataset_target_feature,
            "percentage_to_swap": percentage_to_swap,
            "abrupt_drift_labels_swap": self.abrupt_drift_labels_swap,
            "invert_drift_selection": invert_drift_selection,
        }

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= current_round < self.drift_end_round
        ):
            train_dataloader_rrt = dataloader.load_dataset_from_disk(
                data_type="train_data",
                is_drifted=False,
                create_fedau_dataloader=False,
                **dataloader_props,
            )
            val_dataloader_rrt = dataloader.load_dataset_from_disk(
                data_type="val_data",
                is_drifted=False,
                create_fedau_dataloader=False,
                **dataloader_props,
            )

            if adaptive_fluid_model != "rrt-only":
                train_dataloader_fedau = dataloader.load_dataset_from_disk(
                    data_type="train_data",
                    is_drifted=is_drifted,
                    create_fedau_dataloader=create_fedau_dataloader,
                    **dataloader_props,
                )
                val_dataloader_fedau = dataloader.load_dataset_from_disk(
                    data_type="val_data",
                    is_drifted=is_drifted,
                    create_fedau_dataloader=False,
                    **dataloader_props,
                )

        else:
            train_dataloader = dataloader.load_dataset_from_disk(
                data_type="train_data",
                is_drifted=is_drifted,
                create_fedau_dataloader=create_fedau_dataloader,
                **dataloader_props,
            )
            val_dataloader = dataloader.load_dataset_from_disk(
                data_type="val_data",
                is_drifted=is_drifted,
                create_fedau_dataloader=False,
                **dataloader_props,
            )

        if create_fedau_dataloader and self.mode == "adaptive-fluid":
            train_dataloader_fedau, fedau_dataloader = train_dataloader_fedau
        elif create_fedau_dataloader:
            train_dataloader, fedau_dataloader = train_dataloader

        if self.mode == "adaptive-fluid":
            weights_rrt_index_start = config.get("weights_rrt_index_start", -1)
            weights_rrt_index_end = config.get("weights_rrt_index_end", -1)

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= current_round < self.drift_end_round
        ):
            if weights_rrt_index_start == -1 and weights_rrt_index_end == -1:
                set_weights(self.net_fedau, parameters)
            if weights_rrt_index_start != -1 and weights_rrt_index_end != -1:
                set_weights(
                    self.net_rrt,
                    parameters[weights_rrt_index_start:weights_rrt_index_end],
                )
                set_weights(self.net_fedau, parameters[:weights_rrt_index_start])

            if adaptive_fluid_model != "rrt-only":
                dataset_length = len(train_dataloader_rrt.dataset) + len(
                    train_dataloader_fedau.dataset
                )
            else:
                dataset_length = len(train_dataloader_rrt.dataset)
        elif self.mode == "adaptive-fluid" and current_round == self.drift_end_round:
            set_weights(
                self.net, parameters[weights_rrt_index_start:weights_rrt_index_end]
            )

            dataset_length = len(train_dataloader.dataset)
        else:
            set_weights(self.net, parameters)

            dataset_length = len(train_dataloader.dataset)

        train_props = {
            "epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "dataset_input_feature": self.dataset_input_feature,
            "dataset_target_feature": self.dataset_target_feature,
        }

        if (
            (
                self.mode == "rapid-retraining"
                and current_round >= self.drift_start_round
            )
            or (self.mode == "fluid" and current_round >= self.drift_end_round)
            or (self.mode == "adaptive-fluid" and current_round >= self.drift_end_round)
        ):
            train_results = rapid_train(
                net=self.net,
                trainloader=train_dataloader,
                testloader=val_dataloader,
                batch_size=self.batch_size,
                **train_props,
            )
        elif (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= current_round < self.drift_end_round
        ):
            train_results_rrt = rapid_train(
                net=self.net_rrt,
                trainloader=train_dataloader_rrt,
                testloader=val_dataloader_rrt,
                batch_size=self.batch_size,
                **train_props,
            )
            if adaptive_fluid_model != "rrt-only":
                train_results_fedau = train(
                    net=self.net_fedau,
                    trainloader=train_dataloader_fedau,
                    testloader=val_dataloader_fedau,
                    **train_props,
                )

                train_results = {
                    "train_loss_rrt": train_results_rrt["train_loss"],
                    "train_accuracy_rrt": train_results_rrt["train_accuracy"],
                    "train_loss_fedau": train_results_fedau["train_loss"],
                    "train_accuracy_fedau": train_results_fedau["train_accuracy"],
                }
            else:
                train_results = {
                    "train_loss_rrt": train_results_rrt["train_loss"],
                    "train_accuracy_rrt": train_results_rrt["train_accuracy"],
                }
        else:
            train_results = train(
                net=self.net,
                trainloader=train_dataloader,
                testloader=val_dataloader,
                **train_props,
            )

        train_results.update(
            {
                "client_number": self.client_number,
                "is_drifted": is_drifted,
                "create_fedau_dataloader": create_fedau_dataloader,
            }
        )

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= current_round < self.drift_end_round
        ):
            weights_fedau = get_weights(self.net_fedau)
            weights_rrt = get_weights(self.net_rrt)

            weights = weights_fedau + weights_rrt

            weights_rrt_index_start = len(weights_fedau)
            weights_rrt_index_end = weights_rrt_index_start + len(weights_rrt)

            train_results.update(
                {
                    "weights_rrt_index_start": weights_rrt_index_start,
                    "weights_rrt_index_end": weights_rrt_index_end,
                }
            )
        else:
            weights = get_weights(self.net)

        if create_fedau_dataloader:
            aux_model = Net()

            if (
                self.mode == "adaptive-fluid"
                and weights_rrt_index_start != -1
                and weights_rrt_index_end != -1
            ):
                set_weights(aux_model, parameters[:weights_rrt_index_start])
            else:
                set_weights(aux_model, parameters)

            aux_model.fc3 = nn.Linear(
                aux_model.fc3.in_features, aux_model.fc3.out_features
            )

            if self.mode == "adaptive-fluid":
                val_dataloader = val_dataloader_fedau

            train(
                aux_model,
                fedau_dataloader,
                val_dataloader,
                self.local_epochs,
                self.learning_rate,
                self.device,
                self.dataset_input_feature,
                self.dataset_target_feature,
                0.9,
            )

            aux_last_layer_index = []
            aux_last_layer_array = [
                val.cpu().numpy()
                for idx, (name, val) in enumerate(aux_model.state_dict().items())
                if "fc3" in name and aux_last_layer_index.append(idx) is None
            ]

            train_results.update(
                {
                    "aux_last_layer_weights_index": aux_last_layer_index[0],
                    "aux_last_layer_bias_index": aux_last_layer_index[1],
                }
            )

            weights.append(aux_last_layer_array[0])
            weights.append(aux_last_layer_array[1])

        self.logger.info("train_results %s", train_results)

        return (
            weights,
            dataset_length,
            train_results,
        )

    def evaluate(self, parameters, config):
        current_round = config.get("current_round")
        adaptive_fluid_model = config.get("adaptive_fluid_model", None)

        self.abrupt_drift_labels_swap_modify(current_round)

        is_drifted, percentage_to_swap, _, invert_drift_selection = (
            self._get_drift_config(current_round, adaptive_fluid_model)
        )
        self.logger.info(
            "is_drifted %s, percentage_to_swap %s, invert_drift_selection %s",
            is_drifted,
            percentage_to_swap,
            invert_drift_selection,
        )

        dataloader = DataLoader(
            dataset_input_feature=self.dataset_input_feature,
        )

        eval_dataloader_props = {
            "data_type": "val_data",
            "client_folder_path": self.client_data_folder_path,
            "batch_size": self.batch_size,
            "num_of_batches": self.num_of_batches,
            "dataset_input_feature": self.dataset_input_feature,
            "dataset_target_feature": self.dataset_target_feature,
            "percentage_to_swap": percentage_to_swap,
            "abrupt_drift_labels_swap": self.abrupt_drift_labels_swap,
            "create_fedau_dataloader": False,
            "invert_drift_selection": invert_drift_selection,
        }

        test_props = {
            "device": self.device,
            "dataset_input_feature": self.dataset_input_feature,
            "dataset_target_feature": self.dataset_target_feature,
        }

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= current_round < self.drift_end_round
        ):
            weights_rrt_index_start = config.get("weights_rrt_index_start")
            weights_rrt_index_end = config.get("weights_rrt_index_end")

            set_weights(
                self.net_rrt, parameters[weights_rrt_index_start:weights_rrt_index_end]
            )
            if adaptive_fluid_model != "rrt-only":
                set_weights(
                    self.net_fedau,
                    parameters[:weights_rrt_index_start],
                )
            eval_dataloader_rrt = dataloader.load_dataset_from_disk(
                is_drifted=False,
                **eval_dataloader_props,
            )

            if adaptive_fluid_model != "rrt-only":
                eval_dataloader_fedau = dataloader.load_dataset_from_disk(
                    is_drifted=is_drifted,
                    **eval_dataloader_props,
                )

            eval_loss_rrt, eval_acc_rrt = test(
                net=self.net_rrt,
                testloader=eval_dataloader_rrt,
                **test_props,
            )

            if adaptive_fluid_model != "rrt-only":
                eval_loss_fedau, eval_acc_fedau = test(
                    net=self.net_fedau,
                    testloader=eval_dataloader_fedau,
                    **test_props,
                )

            if adaptive_fluid_model == "fedau":
                eval_loss = eval_loss_fedau
                eval_acc = eval_acc_fedau
            else:
                eval_loss = eval_loss_rrt
                eval_acc = eval_acc_rrt

            evaluate_results = {
                "client_number": self.client_number,
                "loss": eval_loss,
                "accuracy": eval_acc,
                "eval_loss_rrt": eval_loss_rrt,
                "eval_acc_rrt": eval_acc_rrt,
            }

            if adaptive_fluid_model != "rrt-only":
                evaluate_results.update(
                    {
                        "eval_loss_fedau": eval_loss_fedau,
                        "eval_acc_fedau": eval_acc_fedau,
                    }
                )

                eval_dataset_length = len(eval_dataloader_rrt.dataset) + len(
                    eval_dataloader_fedau.dataset
                )
            else:
                eval_dataset_length = len(eval_dataloader_rrt.dataset)

        else:
            set_weights(self.net, parameters)

            eval_dataloader = dataloader.load_dataset_from_disk(
                is_drifted=is_drifted,
                **eval_dataloader_props,
            )

            eval_loss, eval_acc = test(
                net=self.net,
                testloader=eval_dataloader,
                **test_props,
            )

            evaluate_results = {
                "client_number": self.client_number,
                "loss": eval_loss,
                "accuracy": eval_acc,
            }

            eval_dataset_length = len(eval_dataloader.dataset)

        self.logger.info("evaluate_results %s", evaluate_results)

        return (
            eval_loss,
            eval_dataset_length,
            evaluate_results,
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    client_number = context.node_config.get("partition-id")
    batch_size = context.run_config.get("batch-size")
    local_epochs = context.run_config.get("local-epochs")
    learning_rate = context.run_config.get("learning-rate")
    dataset_folder_path = context.run_config.get("dataset-folder-path")
    dataset_input_feature = context.run_config.get("dataset-input-feature")
    dataset_target_feature = context.run_config.get("dataset-target-feature")

    drift_start_round = context.run_config.get("drift-start-round")
    drift_end_round = context.run_config.get("drift-end-round")
    drift_clients = json.loads(str(context.run_config.get("drift-clients")))
    abrupt_drift_labels_swap = json.loads(
        str(context.run_config.get("abrupt-drift-labels-swap"))
    )
    incremental_drift_rounds = json.loads(
        str(context.run_config.get("incremental-drift-rounds"))
    )
    mode = context.run_config.get("mode")
    num_of_batches = context.run_config.get("num-of-batches")
    num_of_clients = context.run_config.get("num-of-clients")

    # Return Client instance
    return FlowerClient(
        client_number,
        batch_size,
        local_epochs,
        learning_rate,
        dataset_folder_path,
        dataset_input_feature,
        dataset_target_feature,
        drift_start_round,
        drift_end_round,
        drift_clients,
        abrupt_drift_labels_swap,
        incremental_drift_rounds,
        mode,
        num_of_batches,
        num_of_clients,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
