import torch


def train_on_batch(net, optimizer, batch, device, scaler=None):
    states, policies, values = batch
    states = states.to(device)
    if device.type == "cuda":
        states = states.contiguous(memory_format=torch.channels_last)
    policies = policies.to(device)
    values = values.to(device)

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32

    optimizer.zero_grad()
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        predicted_policies, predicted_values = net(states)
        value_loss = torch.mean((predicted_values.float() - values) ** 2)
        log_probs = torch.log_softmax(predicted_policies.float(), dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        loss = value_loss + policy_loss

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {
        "loss": float(loss.item()),
        "value_loss": float(value_loss.item()),
        "policy_loss": float(policy_loss.item()),
    }


def build_batch(samples):
    states = torch.stack([torch.as_tensor(sample[0], dtype=torch.float32) for sample in samples], dim=0)
    policies = torch.stack(
        [torch.as_tensor(sample[1], dtype=torch.float32) for sample in samples], dim=0
    )
    values = torch.tensor(
        [[float(sample[2])] for sample in samples], dtype=torch.float32
    )
    return states, policies, values
