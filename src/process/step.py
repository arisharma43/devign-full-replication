import torch
from ..utils.objects import stats


def softmax_accuracy(probs, all_labels):
    labels = all_labels.view_as(probs)
    preds = (probs >= 0.5).float()
    acc = (preds == labels).float().mean()
    return acc


class Step:
    # Performs a step on the loader and returns the result
    def __init__(self, model, loss_function, optimizer, scheduler=None):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, i, x, y):
        logits = self.model(x)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)
        acc = softmax_accuracy(probs, y.float())

        if self.model.training:
            # calculates the gradient
            loss.backward()
            # and performs a parameter update based on it
            self.optimizer.step()
            # clears old gradients from the last step
            self.optimizer.zero_grad()

        # print(f"\tBatch: {i}; Loss: {round(loss.item(), 4)}", end="")
        return stats.Stat(probs.tolist(), loss.item(), acc.item(), y.tolist())

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def scheduler_step(self, metric):
        if self.scheduler is not None:
            self.scheduler.step(metric)

    def current_lr(self):
        if self.optimizer.param_groups:
            return self.optimizer.param_groups[0].get("lr")
        return None
