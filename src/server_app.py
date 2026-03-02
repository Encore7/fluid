import json
import math
import os
from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import (
    Context,
    EvaluateIns,
    FitIns,
    Metrics,
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace

from src.ml_models.cnn import Net
from src.ml_models.utils import get_weights, initialize_model


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [
        float(num_examples) * float(m["accuracy"])
        for num_examples, m in metrics
        if isinstance(m.get("accuracy", None), (int, float))
    ]
    examples = [
        float(num_examples)
        for num_examples, m in metrics
        if isinstance(m.get("accuracy", None), (int, float))
    ]
    return {"accuracy": sum(accuracies) / sum(examples) if examples else 0.0}


class CustomFedAvg(FedAvg):
    def __init__(
        self,
        num_of_clients,
        num_server_rounds,
        plots_folder_path,
        dataset_name,
        incremental_drift_rounds,
        mode,
        drift_start_round,
        drift_end_round,
        adaptive_fluid_blend_decay,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_of_clients = num_of_clients
        self.num_server_rounds = num_server_rounds
        self.plots_folder_path = plots_folder_path
        self.dataset_name = dataset_name
        self.incremental_drift_rounds = incremental_drift_rounds
        self.mode = mode
        self.drift_start_round = drift_start_round
        self.drift_end_round = drift_end_round
        self.adaptive_fluid_blend_decay = adaptive_fluid_blend_decay

        self.client_plot = {}
        self.weights_rrt_index_start = -1
        self.weights_rrt_index_end = -1
        self.adaptive_fluid_model = -1
        self.rrt_blend_weight = -1
        self.fedau_blend_weight = -1

    def _set_metrics(self, client_number, server_round, result, key_name):
        self.client_plot.setdefault(client_number, {})
        self.client_plot[client_number].setdefault(server_round, {})

        self.client_plot[client_number][server_round][f"{key_name}_metrics"] = (
            json.loads(json.dumps(result.metrics, default=str))
        )

    def _is_incremental_drift_start(self, server_round):
        if server_round in map(int, self.incremental_drift_rounds.keys()):
            return True
        return False

    def _is_reinitialize_parameters(self, server_round):
        if (
            self.mode == "retraining"
            or self.mode == "rapid-retraining"
            or self.mode == "adaptive-fluid"
        ) and self.drift_start_round <= server_round < self.drift_end_round:
            return self._is_incremental_drift_start(server_round)
        return False

    def _custom_aggregate(self, results: list[tuple[NDArrays, float]]) -> NDArrays:
        """
        Aggregate model parameters with custom weightages.
        Parameters:
        results: List of tuples, where each tuple contains:
            - NDArrays: Model parameters
            - float: Weightage for this model (e.g., 0.1 for 10%, 0.9 for 90%)
        Returns:
        NDArrays: Aggregated model parameters.
        """
        # Ensure weightages sum up to 1 for valid aggregation
        total_weight = sum(weight for _, weight in results)
        if not np.isclose(total_weight, 1.0):
            raise ValueError("Weightages must sum up to 1.0")
        # Multiply model weights by their respective weightage
        weighted_weights = [
            [layer * weight for layer in weights] for weights, weight in results
        ]
        # Sum up the weighted layers across models
        aggregated_weights: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]
        return aggregated_weights

    def configure_fit(self, server_round, parameters, client_manager):
        # Waiting till all clients are connected
        client_manager.wait_for(self.num_of_clients, timeout=300)

        config: dict[str, Scalar] = {
            "current_round": server_round,
        }

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= server_round <= self.drift_end_round
        ):
            config.update(
                {
                    "weights_rrt_index_start": self.weights_rrt_index_start,
                    "weights_rrt_index_end": self.weights_rrt_index_end,
                    "adaptive_fluid_model": self.adaptive_fluid_model,
                }
            )

        print("fit_ins.config", config)

        if self._is_reinitialize_parameters(server_round):
            net = initialize_model(Net())
            ndarrays = get_weights(net)

            if self.mode == "adaptive-fluid":
                if (
                    self.weights_rrt_index_start != -1
                    and self.weights_rrt_index_end != -1
                ):
                    adaptive_fluid_models_ndarrays = parameters_to_ndarrays(parameters)
                    adaptive_fluid_models_ndarrays[
                        self.weights_rrt_index_start : self.weights_rrt_index_end
                    ] = ndarrays

                    parameters = ndarrays_to_parameters(adaptive_fluid_models_ndarrays)
            else:
                parameters = ndarrays_to_parameters(ndarrays)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Parameters and config
        config: dict[str, Scalar] = {
            "current_round": server_round,
        }

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= server_round < self.drift_end_round
        ):
            config.update(
                {
                    "adaptive_fluid_model": self.adaptive_fluid_model,
                    "weights_rrt_index_start": self.weights_rrt_index_start,
                    "weights_rrt_index_end": self.weights_rrt_index_end,
                }
            )

        print("fit_ins.config", config)

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        aux_models_classifier_layer_list = []
        aux_last_layer_weights_index = -1
        aux_last_layer_bias_index = -1

        for _, fit_res in results:
            print("fit_res.metrics", fit_res.metrics)
            self._set_metrics(
                fit_res.metrics["client_number"], server_round, fit_res, "train"
            )
            if fit_res.metrics["create_fedau_dataloader"] and (
                self.mode == "fedau"
                or self.mode == "fluid"
                or self.mode == "adaptive-fluid"
            ):
                if self.adaptive_fluid_model != "rrt-only":
                    print("Aggregating aux models fit")
                    fit_res_parameters_ndarray = parameters_to_ndarrays(
                        fit_res.parameters
                    )
                    aux_last_layer_weights_index = int(
                        fit_res.metrics["aux_last_layer_weights_index"]
                    )
                    aux_last_layer_bias_index = int(
                        fit_res.metrics["aux_last_layer_bias_index"]
                    )

                    aux_models_classifier_layer_list.append(
                        fit_res_parameters_ndarray[-2:]
                    )
                    fit_res.parameters = ndarrays_to_parameters(
                        fit_res_parameters_ndarray[:-2]
                    )

                if self.mode == "adaptive-fluid":
                    self.weights_rrt_index_start = int(
                        fit_res.metrics["weights_rrt_index_start"]
                    )
                    self.weights_rrt_index_end = int(
                        fit_res.metrics["weights_rrt_index_end"]
                    )

        aggregated_ndarrays = aggregate_inplace(results)

        if aux_models_classifier_layer_list:
            print("Aggregating aux models")
            aux_model_classifier_layer_aggregated = self._custom_aggregate(
                [
                    (
                        aux_models_classifier_layer,
                        1 / len(aux_models_classifier_layer_list),
                    )
                    for aux_models_classifier_layer in aux_models_classifier_layer_list
                ]
            )

            aggregated_ndarrays[
                aux_last_layer_weights_index : aux_last_layer_bias_index + 1
            ] = self._custom_aggregate(
                [
                    (
                        aggregated_ndarrays[
                            aux_last_layer_weights_index : aux_last_layer_bias_index + 1
                        ],
                        0.99,
                    ),
                    (aux_model_classifier_layer_aggregated, 0.01),
                ]
            )

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= server_round < self.drift_end_round
        ):

            def get_blend_parameters():
                return self._custom_aggregate(
                    [
                        (
                            aggregated_ndarrays[: self.weights_rrt_index_start],
                            float(self.fedau_blend_weight),
                        ),
                        (
                            aggregated_ndarrays[
                                self.weights_rrt_index_start : self.weights_rrt_index_end
                            ],
                            float(self.rrt_blend_weight),
                        ),
                    ]
                )

            if self._is_incremental_drift_start(server_round):
                self.adaptive_fluid_model = "fedau"
            elif self.adaptive_fluid_model == "fedau":
                aggregated_ndarrays[: self.weights_rrt_index_start] = (
                    get_blend_parameters()
                )
            elif self.adaptive_fluid_model == "rrt":
                aggregated_ndarrays[
                    self.weights_rrt_index_start : self.weights_rrt_index_end
                ] = get_blend_parameters()

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        return parameters_aggregated, {}

    def aggregate_evaluate(self, server_round, results, failures):

        train_accuracy_rrt = []
        train_accuracy_fedau = []

        for _, eval_res in results:
            print("eval_res.metrics", eval_res.metrics)
            self._set_metrics(
                eval_res.metrics["client_number"], server_round, eval_res, "eval"
            )

            if (
                self.mode == "adaptive-fluid"
                and self.drift_start_round <= server_round < self.drift_end_round
            ):
                train_accuracy_rrt.append(float(eval_res.metrics["eval_acc_rrt"]))
                if self.adaptive_fluid_model != "rrt-only":
                    train_accuracy_fedau.append(
                        float(eval_res.metrics["eval_acc_fedau"])
                    )

        if (
            self.mode == "adaptive-fluid"
            and self.drift_start_round <= server_round < self.drift_end_round
            and self.adaptive_fluid_model != "rrt-only"
        ):
            avg_train_accuracy_rrt = np.mean(train_accuracy_rrt)
            avg_train_accuracy_fedau = np.mean(train_accuracy_fedau)

            def get_blend_weight(accuracy):
                return 1 - (1 - accuracy) ** self.adaptive_fluid_blend_decay

            if avg_train_accuracy_fedau >= avg_train_accuracy_rrt:
                self.adaptive_fluid_model = "fedau"
                self.fedau_blend_weight = get_blend_weight(avg_train_accuracy_fedau)
                self.rrt_blend_weight = 1 - self.fedau_blend_weight
            else:
                self.adaptive_fluid_model = "rrt"
                self.rrt_blend_weight = get_blend_weight(avg_train_accuracy_rrt)
                self.fedau_blend_weight = 1 - self.rrt_blend_weight

            if self.rrt_blend_weight >= 0.9999:
                self.adaptive_fluid_model = "rrt-only"
                self.rrt_blend_weight = 1.0
                self.fedau_blend_weight = 0.0

        if server_round == self.num_server_rounds:
            os.makedirs(self.plots_folder_path, exist_ok=True)

            results_file_path = os.path.join(
                self.plots_folder_path,
                f"{self.dataset_name}_{self.mode}_results.json",
            )

            with open(results_file_path, "w", encoding="utf-8") as file:
                json.dump(self.client_plot, file)

        return super().aggregate_evaluate(server_round, results, failures)


def server_fn(context: Context):
    # Initialize model parameters
    fraction_evaluate = context.run_config.get("fraction-evaluate")
    num_of_clients = context.run_config.get("num-of-clients")
    num_server_rounds = context.run_config.get("num-server-rounds")
    plots_folder_path = context.run_config.get("plots-folder-path")
    dataset_name = context.run_config.get("dataset-name")
    incremental_drift_rounds = json.loads(
        str(context.run_config.get("incremental-drift-rounds"))
    )
    mode = context.run_config.get("mode")
    drift_start_round = context.run_config.get("drift-start-round")
    drift_end_round = context.run_config.get("drift-end-round")
    adaptive_fluid_blend_decay = context.run_config.get("adaptive-fluid-blend-decay")

    # Initialize the model with dummy data to setup lazy modules
    net = initialize_model(Net())
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = CustomFedAvg(
        fraction_evaluate=fraction_evaluate,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        num_of_clients=num_of_clients,
        num_server_rounds=num_server_rounds,
        plots_folder_path=plots_folder_path,
        dataset_name=dataset_name,
        incremental_drift_rounds=incremental_drift_rounds,
        mode=mode,
        drift_start_round=drift_start_round,
        drift_end_round=drift_end_round,
        adaptive_fluid_blend_decay=adaptive_fluid_blend_decay,
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
