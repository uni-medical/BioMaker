# 🧪 Optuna Integration Guide (Short)

Use Optuna to tune hyperparameters in existing PyTorch pipelines.


## ⚙️ 1. Define Reusable Functions

### DataLoader
```python
def get_loaders(batch_size):
    ...
    return train_loader, val_loader
```

### Model
```python
def define_model(model_name, num_classes, in_chans):
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=in_chans).to(device)
```


## 🎯 2. Define `objective(trial)`

```python
def objective(trial):
    model_name = trial.suggest_categorical("model_name", [...])
    lr = trial.suggest_float("lr", 1e-5, 1e-2)
    ...
    model = define_model(...)
    optimizer = ...
    for epoch in range(10):
        train(...)
        val_auc = evaluate(...)
        trial.report(val_auc, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_auc
```

## 🚀 3. Run the Study

```python
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(objective, n_trials=30, timeout=1800)
print("Best Hyperparameters:", study.best_trial.params)
```


## 🏁 4. Final Training

Now use the best trial config for a full training run:

```python
best_params = study.best_trial.params

train_loader, val_loader = get_loaders(best_params["batch_size"])
model = define_model(best_params["model_name"], num_classes=5, in_chans=3)

if best_params["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9, weight_decay=best_params["weight_decay"])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

```
## Full example:
```
# Define MONAI transforms with data augmentation for training
train_transforms = ...
val_transforms = ...


# Define custom Dataset
class CustomDataset(Dataset):
...


# Create datasets and dataloaders
...

# Define model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2)
    num_epochs = trial.suggest_int("num_epochs", 1, 10)

    model = timm.create_model("densenet121", pretrained=True, num_classes=5, in_chans=3).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate
        model.eval()
        ...

        # Calculate validation AUC
        val_auc = roc_auc_score(val_labels, np.eye(5)[val_preds], multi_class="ovr")
        trial.report(val_auc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        scheduler.step(val_auc)

    return val_auc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

print("Best Hyperparameters:", study.best_trial.params)

# Train final model with best hyperparameters
best_params = study.best_trial.params
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3
)

for epoch in range(best_params["num_epochs"]):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validate
    ...

    # Calculate validation AUC
    ...

# Predict on test data
...

# Save predictions to submission.csv
...
```