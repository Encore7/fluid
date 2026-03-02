from collections import OrderedDict

import torch

from src.scripts.helper import metadata


def initialize_model(net):
    """Initialize lazy modules by running a forward pass with dummy data."""
    # Create dummy input based on metadata
    dummy_input = torch.randn(
        1, metadata["num_channels"], metadata["image_height"], metadata["image_width"]
    )
    net.eval()
    with torch.no_grad():
        _ = net(dummy_input)
    return net


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(
    net,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    dataset_input_feature,
    dataset_target_feature,
    momentum=0.0,
):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            optimizer.zero_grad()
            criterion(net(images), labels).backward()
            optimizer.step()

    train_loss, train_acc = test(
        net, testloader, device, dataset_input_feature, dataset_target_feature
    )

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
    }
    return results


def get_weighed_sum(current_grad, all_gradients, beta, t, param, power=1):
    weighed_sum = 0
    for i in range(t):
        if t == i + 1:
            weighed_sum += (beta ** (t - (i + 1))) * (current_grad**power)
        else:
            weighed_sum += (beta ** (t - (i + 1))) * (all_gradients[i][param] ** power)

    return weighed_sum


def rapid_train(
    net,
    trainloader,
    testloader,
    epochs,
    learning_rate,
    device,
    batch_size,
    dataset_input_feature,
    dataset_target_feature,
):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.005

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()

    for _ in range(epochs):
        all_gradients = []
        for batch_idx, batch in enumerate(trainloader):
            t = batch_idx + 1
            batch_gradients = {}
            images = batch[dataset_input_feature]
            labels = batch[dataset_target_feature]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()

            # Square all gradients before optimizer.step()
            for param in net.parameters():
                if param.grad is not None:
                    batch_gradients[param] = param.grad.clone().detach()

                    weighed_sum_beta1 = get_weighed_sum(
                        param.grad, all_gradients, beta1, t, param
                    )
                    weighed_sum_beta2 = get_weighed_sum(
                        param.grad, all_gradients, beta1, t, param, 4
                    )

                    m_t = ((1 - beta1) * weighed_sum_beta1) / (1 - beta1**t)
                    v_t = torch.sqrt(((1 - beta2) * weighed_sum_beta2) / (1 - beta2**t))

                    param.data -= ((m_t) / (v_t + epsilon)) / batch_size

            all_gradients.append(batch_gradients)

    train_loss, train_acc = test(
        net, testloader, device, dataset_input_feature, dataset_target_feature
    )

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
    }
    return results


def test(
    net,
    testloader,
    device,
    dataset_input_feature,
    dataset_target_feature,
):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[dataset_input_feature].to(device)
            labels = batch[dataset_target_feature].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader.dataset)
    return loss, accuracy
