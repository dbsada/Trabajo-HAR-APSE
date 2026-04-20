import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def make_plotly_graph(df, height, width):
    labels = { 1: "Skip", 2: "Twist", 3: "Jump", 4: "Step" }
    axis_colors = {"X": "#1f77b4", "Y": "#ff7f0e", "Z": "#2ca02c"}
    grouped = { lbl: df[df['META']['LABEL'] == lbl].reset_index(drop=True) for lbl in labels.keys()} 
    max_len = max(len(df) for df in grouped.values())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[labels[lbl] for lbl in labels.keys()],
        shared_xaxes=True,
        shared_yaxes=True
    )

    fig.update_xaxes(range=[0, max(len(row["X"]) for df in grouped.values() for _, row in df.iterrows())])
    fig.update_yaxes(range=[
        min(min(min(row["X"]), min(row["Y"]), min(row["Z"])) for df in grouped.values() for _, row in df.iterrows()),
        max(max(max(row["X"]), max(row["Y"]), max(row["Z"])) for df in grouped.values() for _, row in df.iterrows())
    ])

    trace_map = [] 
    for idx, lbl in enumerate(labels.keys()): 
        r = idx // 2 + 1 
        c = idx % 2 + 1 
        for axis_name in ["X", "Y", "Z"]: 
            fig.add_trace( 
                go.Scatter(y=[], mode="lines", name=axis_name, showlegend=(idx == 0), line={"color": axis_colors[axis_name]}), 
                row=r, 
                col=c 
            ) 
            trace_map.append((r, c))

    frames = [] 

    for i in range(max_len): 
        frame_data = [] 

        for lbl in labels.keys(): 
            df = grouped[lbl] 

            if i < len(df): 
                row = df.iloc[i] 
                frame_data.extend([ 
                    go.Scatter(y=row["X"], line={"color": axis_colors["X"]}), 
                    go.Scatter(y=row["Y"], line={"color": axis_colors["Y"]}), 
                    go.Scatter(y=row["Z"], line={"color": axis_colors["Z"]}), 
                ]) 
            else: 
                frame_data.extend([ 
                    go.Scatter(y=[], line={"color": axis_colors["X"]}), 
                    go.Scatter(y=[], line={"color": axis_colors["Y"]}), 
                    go.Scatter(y=[], line={"color": axis_colors["Z"]}), 
                ]) 

        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.frames = frames 
            
    for t_idx, trace in enumerate(frames[0].data): 
        fig.data[t_idx].y = trace.y 
        
    sliders = [{ 
        "steps": [ 
            { 
                "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], 
                "label": str(i), 
                "method": "animate", 
            } for i in range(max_len)], 
        "currentvalue": {"prefix": "Row: "} 
    }] 

    fig.update_layout(title="Sensor Data by Label", title_x=0.5, sliders=sliders, height=height, width=width)
    fig.show()

def make_plotly_graph_height(df_height, labels_map, max_per_label=20, height=400, width=900):
    value_cols = [c for c in df_height.columns if c != "LABEL"]

    # Tomamos hasta max_per_label por clase
    by_label = {
        lbl: df_height[df_height["LABEL"] == lbl].head(max_per_label).reset_index(drop=True)
        for lbl in labels_map.keys()
    }
    n_steps = max((len(v) for v in by_label.values()), default=0)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[labels_map[lbl] for lbl in labels_map.keys()]
    )

    ordered_labels = list(labels_map.keys())

    # Estado inicial (muestra 1)
    for i, lbl in enumerate(ordered_labels):
        r, c = divmod(i, 2)
        r += 1
        c += 1
        s = by_label[lbl]
        y0 = s.loc[0, value_cols].to_numpy() if len(s) > 0 else [None] * len(value_cols)

        fig.add_trace(
            go.Scatter(y=y0, mode="lines", name=f"{labels_map[lbl]} - height", showlegend=False),
            row=r, col=c
        )

    # Slider: cambia la muestra en los 4 subplots a la vez
    steps = []
    for k in range(n_steps):
        ys = []
        for lbl in ordered_labels:
            s = by_label[lbl]
            ys.append(s.loc[k, value_cols].to_numpy() if k < len(s) else [None] * len(value_cols))

        steps.append(
            dict(
                method="restyle",
                args=[{"y": ys}],
                label=str(k + 1)
            )
        )

    fig.update_layout(
        title="Señal de altura relativa por clase",
        height=height,
        width=width,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Muestra: "},
            pad={"t": 30},
            steps=steps
        )]
    )
    fig.update_xaxes(title_text="Tiempo (muestras)")
    fig.update_yaxes(title_text="Height")

    fig.show()

def apply_spec_augment(spec, time_mask_param=None, freq_mask_param=None, num_time_masks=1, num_freq_masks=1):
    """
    SpecAugment simple para tensores [C, H, W].
    - Time masking sobre la dimension W (2)
    - Frequency masking sobre la dimension H (1)
    """
    augmented = spec.clone()
    _, freq_bins, time_bins = augmented.shape

    time_mask_param = max(1, int(time_bins * 0.10)) if time_mask_param is None else min(time_bins, time_mask_param)
    freq_mask_param = max(1, int(freq_bins * 0.10)) if freq_mask_param is None else min(freq_bins, freq_mask_param)

    for _ in range(min(num_time_masks, 2)):
        mask_width = int(torch.randint(1, time_mask_param + 1, (1,), device=augmented.device).item())
        if time_bins > mask_width:
            start = int(torch.randint(0, time_bins - mask_width + 1, (1,), device=augmented.device).item())
            augmented[:, :, start:start + mask_width] = 0

    for _ in range(min(num_freq_masks, 2)):
        mask_height = int(torch.randint(1, freq_mask_param + 1, (1,), device=augmented.device).item())
        if freq_bins > mask_height:
            start = int(torch.randint(0, freq_bins - mask_height + 1, (1,), device=augmented.device).item())
            augmented[:, start:start + mask_height, :] = 0

    return augmented

class ActivityDataset(Dataset):
    def __init__(self, df, channel_fn, cache_dir=None, precompute=False, is_training=False):
        self.df = df.reset_index(drop=True)
        self.channel_fn = channel_fn
        self.cache_dir = cache_dir
        self.is_training = is_training

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            if precompute:
                self.precompute_missing()

    def __len__(self):
        return len(self.df)

    def _cache_path(self, idx):
        return os.path.join(self.cache_dir, f"sample_{idx:06d}.pt")

    def precompute_missing(self):
        missing = [idx for idx in range(len(self.df)) if not os.path.exists(self._cache_path(idx))]
        if len(missing) == 0:
            print(f"Cache listo en {self.cache_dir} ({len(self.df)} muestras)")
            return

        for idx in tqdm(missing, desc=f"Precomputando {self.cache_dir}", leave=False):
            row = self.df.iloc[idx]
            x_data = self.channel_fn(row)
            x_tensor = torch.from_numpy(x_data).float()
            torch.save(x_tensor, self._cache_path(idx))

        print(f"Cache generado en {self.cache_dir}: {len(missing)} nuevos archivos")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.cache_dir is not None:
            cache_path = self._cache_path(idx)
            if os.path.exists(cache_path):
                x_tensor = torch.load(cache_path, map_location="cpu")
            else:
                x_data = self.channel_fn(row)
                x_tensor = torch.from_numpy(x_data).float()
                torch.save(x_tensor, cache_path)
        else:
            x_data = self.channel_fn(row)
            x_tensor = torch.from_numpy(x_data).float()

        x_tensor = x_tensor.clone()
        if self.is_training:
            x_tensor = apply_spec_augment(x_tensor)

        y_label = torch.tensor(int(row[("META", "LABEL")]) - 1, dtype=torch.long)
        return x_tensor, y_label

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out

class ActivityCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.time_stem = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.time_block1 = ResidualBlock(16, 16, stride=1)
        self.time_block2 = ResidualBlock(16, 24, stride=2)

        self.freq_stem = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.freq_block1 = ResidualBlock(16, 16, stride=1)
        self.freq_block2 = ResidualBlock(16, 24, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Linear(48, 32),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def _forward_branch(self, x, stem, b1, b2):
        x = stem(x)
        x = b1(x)
        x = b2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x_time = x[:, :6, :, :]
        x_freq = x[:, 6:, :, :]

        feat_time = self._forward_branch(x_time, self.time_stem, self.time_block1, self.time_block2)
        feat_freq = self._forward_branch(x_freq, self.freq_stem, self.freq_block1, self.freq_block2)

        feat = torch.cat([feat_time, feat_freq], dim=1)
        return self.head(feat)

def train_model(model, train_loader, criterion, optimizer, device, show_batch_progress=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        train_loader,
        desc="Entrenando",
        leave=False,
        colour="blue",
        disable=not show_batch_progress,
    )

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if show_batch_progress:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.2f}%",
            })

    return running_loss / len(train_loader), 100. * correct / total

def validate_model(model, val_loader, criterion, device, show_batch_progress=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        val_loader,
        desc="Validando",
        leave=False,
        colour="blue",
        disable=not show_batch_progress,
    )

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if show_batch_progress:
                pbar.set_postfix({
                    "v_loss": f"{loss.item():.4f}",
                    "v_acc": f"{100. * correct / total:.2f}%",
                })

    return running_loss / len(val_loader), 100. * correct / total

def run_loso_validation(df_full, generate_channels_fn, device):
    user_ids = df_full[('META', 'USER_ID')].unique()
    user_ids = sorted([int(u) for u in user_ids])
    num_subjects = len(user_ids)

    loso_results = []
    loso_histories = []
    all_loso_epochs = []

    outer_pbar = tqdm(user_ids, total=num_subjects, desc="LOSO sujetos", position=0, leave=True)

    for iteration, test_subject_id in enumerate(outer_pbar, 1):
        test_mask = (df_full[("META", "USER_ID")] == test_subject_id).values
        train_mask = ~test_mask

        loso_train_df = df_full[train_mask].reset_index(drop=True)
        loso_test_df = df_full[test_mask].reset_index(drop=True)

        loso_subject_dir = f"artifacts/loso_cache/subject_{test_subject_id:02d}"
        loso_train_cache = f"{loso_subject_dir}/train"
        loso_test_cache = f"{loso_subject_dir}/test"

        os.makedirs(loso_train_cache, exist_ok=True)
        os.makedirs(loso_test_cache, exist_ok=True)

        model_loso = ActivityCNN(num_classes=4).to(device)
        criterion_loso = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer_loso = torch.optim.Adam(model_loso.parameters(), lr=0.0005, weight_decay=1e-3)
        scheduler_loso = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_loso, "min", patience=4, factor=0.7)

        train_ds_loso = ActivityDataset(
            loso_train_df,
            channel_fn=generate_channels_fn,
            cache_dir=loso_train_cache,
            precompute=False,
            is_training=True,
        )
        test_ds_loso = ActivityDataset(
            loso_test_df,
            channel_fn=generate_channels_fn,
            cache_dir=loso_test_cache,
            precompute=False,
            is_training=False,
        )

        train_loader_loso = DataLoader(train_ds_loso, batch_size=32, shuffle=True, num_workers=0)
        test_loader_loso = DataLoader(test_ds_loso, batch_size=32, shuffle=False, num_workers=0)

        best_val_loss_loso = float("inf")
        patience_loso = 15
        patience_counter_loso = 0
        best_test_acc_loso = 0.0
        best_model_path = f"{loso_subject_dir}/best_model.pth"

        history_loso = {
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "lr": [],
        }

        max_epochs_loso = 60
        epoch_pbar = tqdm(
            range(max_epochs_loso),
            total=max_epochs_loso,
            desc=f"Sujeto {test_subject_id:02d} epocas",
            position=1,
            leave=False,
        )

        for epoch_loso in epoch_pbar:
            train_loss_loso, train_acc_loso = train_model(
                model_loso, train_loader_loso, criterion_loso, optimizer_loso, device, show_batch_progress=False
            )
            test_loss_loso, test_acc_loso = validate_model(
                model_loso, test_loader_loso, criterion_loso, device, show_batch_progress=False
            )
            scheduler_loso.step(test_loss_loso)

            lr_actual_loso = optimizer_loso.param_groups[0]["lr"]

            history_loso["epoch"].append(epoch_loso)
            history_loso["train_loss"].append(train_loss_loso)
            history_loso["test_loss"].append(test_loss_loso)
            history_loso["train_acc"].append(train_acc_loso)
            history_loso["test_acc"].append(test_acc_loso)
            history_loso["lr"].append(lr_actual_loso)

            if test_loss_loso < best_val_loss_loso:
                best_val_loss_loso = test_loss_loso
                patience_counter_loso = 0
                best_test_acc_loso = test_acc_loso
                torch.save(model_loso.state_dict(), best_model_path)
            else:
                patience_counter_loso += 1

            epoch_pbar.set_postfix({
                "tr_acc": f"{train_acc_loso:.1f}%",
                "te_acc": f"{test_acc_loso:.1f}%",
                "best": f"{best_test_acc_loso:.1f}%",
                "pat": f"{patience_counter_loso}/{patience_loso}",
            })

            if patience_counter_loso >= patience_loso:
                model_loso.load_state_dict(torch.load(best_model_path))
                break

        epoch_pbar.close()

        loso_results.append({"subject_id": test_subject_id, "accuracy": best_test_acc_loso})
        loso_histories.append({"subject_id": test_subject_id, "history": history_loso})
        all_loso_epochs.append({"subject_id": test_subject_id, "history": history_loso})

        current_mean = np.mean([r["accuracy"] for r in loso_results])
        outer_pbar.set_postfix({
            "ultimo": f"S{test_subject_id:02d}={best_test_acc_loso:.1f}%",
            "media": f"{current_mean:.1f}%",
        })

    outer_pbar.close()
    return loso_results, loso_histories, all_loso_epochs


def compute_loso_epoch_summary(all_loso_epochs):
    """Build a per-epoch summary across subjects without side effects."""
    if len(all_loso_epochs) == 0:
        return pd.DataFrame(
            columns=[
                "Epoch",
                "Avg_Train_Acc",
                "Avg_Test_Acc",
                "Avg_Train_Loss",
                "Avg_Test_Loss",
                "Std_Test_Acc",
            ]
        )

    max_epochs_all = max(len(h["history"]["epoch"]) for h in all_loso_epochs)
    epoch_range = np.arange(1, max_epochs_all + 1)

    avg_train_accs, avg_test_accs = [], []
    avg_train_losses, avg_test_losses = [], []
    std_test_accs = []

    for epoch_idx in range(max_epochs_all):
        train_accs, test_accs = [], []
        train_losses, test_losses = [], []

        for subject_history in all_loso_epochs:
            hist = subject_history["history"]
            if epoch_idx < len(hist["epoch"]):
                train_accs.append(hist["train_acc"][epoch_idx])
                test_accs.append(hist["test_acc"][epoch_idx])
                train_losses.append(hist["train_loss"][epoch_idx])
                test_losses.append(hist["test_loss"][epoch_idx])

        avg_train_accs.append(np.mean(train_accs) if train_accs else np.nan)
        avg_test_accs.append(np.mean(test_accs) if test_accs else np.nan)
        avg_train_losses.append(np.mean(train_losses) if train_losses else np.nan)
        avg_test_losses.append(np.mean(test_losses) if test_losses else np.nan)
        std_test_accs.append(np.std(test_accs) if test_accs else np.nan)

    return pd.DataFrame(
        {
            "Epoch": epoch_range,
            "Avg_Train_Acc": avg_train_accs,
            "Avg_Test_Acc": avg_test_accs,
            "Avg_Train_Loss": avg_train_losses,
            "Avg_Test_Loss": avg_test_losses,
            "Std_Test_Acc": std_test_accs,
        }
    )


def compute_loso_subject_ranking(loso_results):
    """Return per-subject LOSO performance sorted from best to worst."""
    if len(loso_results) == 0:
        return pd.DataFrame(columns=["subject_id", "accuracy"])

    df_results = pd.DataFrame(loso_results)
    return df_results.sort_values("accuracy", ascending=False).reset_index(drop=True)


def compute_loso_global_metrics(loso_results, confidence_level=0.95):
    """Compute global LOSO accuracy stats and confidence interval."""
    df_results = compute_loso_subject_ranking(loso_results)
    if df_results.empty:
        return {
            "mean_acc": np.nan,
            "std_acc": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "margin_error": np.nan,
            "num_subjects": 0,
        }

    accuracies = df_results["accuracy"].to_numpy(dtype=float)
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    if len(accuracies) > 1:
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha / 2, len(accuracies) - 1)
        margin_error = float(t_value * (std_acc / np.sqrt(len(accuracies))))
        ci_lower = mean_acc - margin_error
        ci_upper = mean_acc + margin_error
    else:
        margin_error = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    return {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_error": margin_error,
        "num_subjects": int(len(accuracies)),
    }


def compute_loso_generalization_gap(all_loso_epochs):
    """Compute train-test generalization gap per epoch and summary points."""
    epoch_summary = compute_loso_epoch_summary(all_loso_epochs)
    if epoch_summary.empty:
        return {
            "epoch_range": np.array([]),
            "gap": np.array([]),
            "max_gap": np.nan,
            "max_gap_epoch": np.nan,
            "final_gap": np.nan,
        }

    gap = (epoch_summary["Avg_Train_Acc"] - epoch_summary["Avg_Test_Acc"]).to_numpy(dtype=float)
    epoch_range = epoch_summary["Epoch"].to_numpy(dtype=int)

    max_gap_idx = int(np.nanargmax(gap))
    max_gap = float(gap[max_gap_idx])
    max_gap_epoch = int(epoch_range[max_gap_idx])
    final_gap = float(gap[-1])

    return {
        "epoch_range": epoch_range,
        "gap": gap,
        "max_gap": max_gap,
        "max_gap_epoch": max_gap_epoch,
        "final_gap": final_gap,
    }


def compute_loso_saturation_epoch(epoch_summary, improvement_threshold=0.01):
    """Estimate the first epoch where avg test loss changes less than a threshold."""
    if epoch_summary is None or len(epoch_summary) == 0:
        return np.nan

    avg_losses = epoch_summary["Avg_Test_Loss"].to_numpy(dtype=float)
    epoch_nums = epoch_summary["Epoch"].to_numpy(dtype=int)

    if len(avg_losses) < 2:
        return int(epoch_nums[0]) if len(epoch_nums) else np.nan

    gradients = np.diff(avg_losses)
    epochs_with_gradient = epoch_nums[1:]
    saturated_epochs = epochs_with_gradient[np.abs(gradients) < improvement_threshold]

    return int(saturated_epochs[0]) if len(saturated_epochs) > 0 else int(epoch_nums[-1])


def compute_loso_top_subjects(loso_results, top_k=3):
    """Return top-k subjects sorted by LOSO accuracy."""
    ranked = compute_loso_subject_ranking(loso_results)
    return ranked.head(top_k).reset_index(drop=True)

def analyze_loso_convergence(all_loso_epochs, loso_results):
    max_epochs_all = max([len(h["history"]["epoch"]) for h in all_loso_epochs])
    print(f"Épocas máximas alcanzadas: {max_epochs_all}")

    convergence_data = []

    for epoch_idx in range(max_epochs_all):
        epoch_row = {"Epoch": epoch_idx + 1}
        
        test_accs_at_epoch = []
        test_losses_at_epoch = []
        
        for subject_history in all_loso_epochs:
            subject_id = subject_history["subject_id"]
            hist = subject_history["history"]
            
            if epoch_idx < len(hist["epoch"]):
                epoch_row[f"Subject_{subject_id}_Acc"] = hist["test_acc"][epoch_idx]
                epoch_row[f"Subject_{subject_id}_Loss"] = hist["test_loss"][epoch_idx]
                test_accs_at_epoch.append(hist["test_acc"][epoch_idx])
                test_losses_at_epoch.append(hist["test_loss"][epoch_idx])
        
        epoch_row["Avg_Test_Acc"] = np.mean(test_accs_at_epoch) if test_accs_at_epoch else np.nan
        epoch_row["Avg_Test_Loss"] = np.mean(test_losses_at_epoch) if test_losses_at_epoch else np.nan
        epoch_row["Std_Test_Acc"] = np.std(test_accs_at_epoch) if test_accs_at_epoch else np.nan
        
        convergence_data.append(epoch_row)

    df_convergence = pd.DataFrame(convergence_data)

    print("Matriz de convergencia (primeras 10 épocas):")
    print(df_convergence[["Epoch", "Avg_Test_Acc", "Std_Test_Acc", "Avg_Test_Loss"]].head(10).to_string(index=False))

    avg_losses = df_convergence["Avg_Test_Loss"].values
    epoch_nums = df_convergence["Epoch"].values

    gradients = np.diff(avg_losses)
    epochs_with_gradient = epoch_nums[1:]

    improvement_threshold = 0.01
    saturated_epochs = epochs_with_gradient[np.abs(gradients) < improvement_threshold]

    if len(saturated_epochs) > 0:
        saturation_epoch = saturated_epochs[0]
        print(f"Punto de saturación (val loss): época {saturation_epoch}")
    else:
        saturation_epoch = max_epochs_all
        print(f"No hay punto de saturación claro en {max_epochs_all} épocas")

    df_results = pd.DataFrame(loso_results)
    df_results = df_results.sort_values("accuracy", ascending=False).reset_index(drop=True)

    print("" + df_results.to_string(index=False))

    accuracies = df_results["accuracy"].values
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    median_acc = np.median(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    print(f"Sujetos evaluados: {len(loso_results)}")
    print(f"Media: {mean_acc:.2f}%")
    print(f"Desv. est.: {std_acc:.2f}%")
    print(f"Mediana: {median_acc:.2f}%")
    print(f"Mínimo: {min_acc:.2f}%")
    print(f"Máximo: {max_acc:.2f}%")

    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, len(accuracies) - 1)
    margin_error = t_value * (std_acc / np.sqrt(len(accuracies)))
    ci_lower = mean_acc - margin_error
    ci_upper = mean_acc + margin_error

    print(f"IC 95%: [{ci_lower:.2f}% , {ci_upper:.2f}%]")
    print(f"Margen de error: ±{margin_error:.2f}%")

    excel_path = "artifacts/loso_results.xlsx"
    os.makedirs("artifacts", exist_ok=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Per_Subject", index=False)
        df_convergence.to_excel(writer, sheet_name="Convergence_Matrix", index=False)
        
        summary_stats = pd.DataFrame({
            "Metric": ["Mean Accuracy", "Std Dev", "Median", "Min", "Max", "CI Lower (95%)", "CI Upper (95%)", "Saturation Epoch"],
            "Value": [f"{mean_acc:.2f}%", f"{std_acc:.2f}%", f"{median_acc:.2f}%", 
                      f"{min_acc:.2f}%", f"{max_acc:.2f}%", f"{ci_lower:.2f}%", f"{ci_upper:.2f}%", f"{saturation_epoch}"]
        })
        summary_stats.to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ Resultados exportados a: {excel_path}")

    stats_dict = {
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "median_acc": median_acc,
        "min_acc": min_acc,
        "max_acc": max_acc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "excel_path": excel_path
    }
    
    return df_convergence, df_results, saturation_epoch, stats_dict

def plot_loso_analysis(all_loso_epochs, df_results, saturation_epoch, mean_acc):
    max_epochs_all = max([len(h["history"]["epoch"]) for h in all_loso_epochs])
    avg_train_accs, avg_test_accs, avg_train_losses, avg_test_losses = [], [], [], []

    for epoch_idx in range(max_epochs_all):
        train_accs, test_accs, train_losses, test_losses = [], [], [], []
        
        for subject_history in all_loso_epochs:
            hist = subject_history["history"]
            if epoch_idx < len(hist["epoch"]):
                train_accs.append(hist["train_acc"][epoch_idx])
                test_accs.append(hist["test_acc"][epoch_idx])
                train_losses.append(hist["train_loss"][epoch_idx])
                test_losses.append(hist["test_loss"][epoch_idx])
        
        avg_train_accs.append(np.mean(train_accs) if train_accs else np.nan)
        avg_test_accs.append(np.mean(test_accs) if test_accs else np.nan)
        avg_train_losses.append(np.mean(train_losses) if train_losses else np.nan)
        avg_test_losses.append(np.mean(test_losses) if test_losses else np.nan)

    epoch_range = np.arange(1, max_epochs_all + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    axs[0, 0].plot(epoch_range, avg_train_accs, label="Train Acc (Promedio)", marker="o", markersize=3, linewidth=2)
    axs[0, 0].plot(epoch_range, avg_test_accs, label="Test Acc (Promedio)", marker="s", markersize=3, linewidth=2)
    axs[0, 0].axvline(saturation_epoch, color="red", linestyle="--", alpha=0.7, label=f"Punto de Saturación (Época {saturation_epoch})")
    axs[0, 0].set_title("Accuracy Promedio: Train vs Test (LOSO)", fontsize=12, fontweight="bold")
    axs[0, 0].set_xlabel("Época")
    axs[0, 0].set_ylabel("Accuracy (%)")
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].grid(alpha=0.3)

    axs[0, 1].plot(epoch_range, avg_train_losses, label="Train Loss (Promedio)", marker="o", markersize=3, linewidth=2)
    axs[0, 1].plot(epoch_range, avg_test_losses, label="Test Loss (Promedio)", marker="s", markersize=3, linewidth=2)
    axs[0, 1].axvline(saturation_epoch, color="red", linestyle="--", alpha=0.7, label=f"Punto de Saturación (Época {saturation_epoch})")
    axs[0, 1].set_title("Loss Promedio: Train vs Test (LOSO)", fontsize=12, fontweight="bold")
    axs[0, 1].set_xlabel("Época")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(alpha=0.3)

    axs[1, 0].barh(range(len(df_results)), df_results["accuracy"].values, color="steelblue", alpha=0.8)
    axs[1, 0].axvline(mean_acc, color="red", linestyle="--", linewidth=2, label=f"Media: {mean_acc:.2f}%")
    axs[1, 0].set_yticks(range(len(df_results)))
    axs[1, 0].set_yticklabels([f"Subject {s}" for s in df_results["subject_id"].values], fontsize=8)
    axs[1, 0].set_xlabel("Accuracy (%)")
    axs[1, 0].set_title("Mejor Accuracy por Sujeto (LOSO)", fontsize=12, fontweight="bold")
    axs[1, 0].legend()
    axs[1, 0].grid(alpha=0.3, axis="x")

    std_test_accs = []
    for epoch_idx in range(max_epochs_all):
        test_accs = []
        for subject_history in all_loso_epochs:
            hist = subject_history["history"]
            if epoch_idx < len(hist["epoch"]):
                test_accs.append(hist["test_acc"][epoch_idx])
        std_test_accs.append(np.std(test_accs) if test_accs else 0)

    std_test_accs = np.array(std_test_accs)
    axs[1, 1].plot(epoch_range, avg_test_accs, label="Test Acc Promedio", marker="o", markersize=3, linewidth=2, color="green")
    axs[1, 1].fill_between(epoch_range, 
                            np.array(avg_test_accs) - std_test_accs,
                            np.array(avg_test_accs) + std_test_accs,
                            alpha=0.2, color="green", label="±1 Std Dev")
    axs[1, 1].axvline(saturation_epoch, color="red", linestyle="--", alpha=0.7, label=f"Saturación (Época {saturation_epoch})")
    axs[1, 1].set_title("Convergencia Test Acc (Promedio ± Desv. Est.)", fontsize=12, fontweight="bold")
    axs[1, 1].set_xlabel("Época")
    axs[1, 1].set_ylabel("Accuracy (%)")
    axs[1, 1].legend()
    axs[1, 1].grid(alpha=0.3)

    plt.savefig("artifacts/loso_analysis.png", dpi=300, bbox_inches="tight")
    print("Gráfico guardado: artifacts/loso_analysis.png")
    plt.show()

    df_epoch_summary = pd.DataFrame({
        "Epoch": epoch_range,
        "Avg_Train_Acc": avg_train_accs,
        "Avg_Test_Acc": avg_test_accs,
        "Avg_Train_Loss": avg_train_losses,
        "Avg_Test_Loss": avg_test_losses,
        "Std_Test_Acc": std_test_accs
    })

    print(df_epoch_summary.head(20).to_string(index=False))

    epoch_summary_path = "artifacts/loso_epoch_summary.xlsx"
    df_epoch_summary.to_excel(epoch_summary_path, sheet_name="Epoch_Summary", index=False)

    print(f"Tabla de épocas exportada a: {epoch_summary_path}")


def analyze_loso_overfitting_and_summary(all_loso_epochs, num_subjects, df_results, df_convergence, saturation_epoch, stats):
    max_epochs_all = max([len(h["history"]["epoch"]) for h in all_loso_epochs])
    generalization_gaps = []

    for epoch_idx in range(max_epochs_all):
        train_accs = []
        test_accs = []
        
        for subject_history in all_loso_epochs:
            hist = subject_history["history"]
            if epoch_idx < len(hist["epoch"]):
                train_accs.append(hist["train_acc"][epoch_idx])
                test_accs.append(hist["test_acc"][epoch_idx])
        
        gap = np.mean(train_accs) - np.mean(test_accs) if (train_accs and test_accs) else np.nan
        generalization_gaps.append(gap)

    max_gap = np.nanmax(generalization_gaps)
    max_gap_epoch = np.nanargmax(generalization_gaps) + 1
    epoch_range = np.arange(1, max_epochs_all + 1)

    print(f"Brecha máxima (train - test): {max_gap:.2f}% en época {max_gap_epoch}")

    final_gap = generalization_gaps[-1]
    print(f"Brecha final: {final_gap:.2f}%")

    if max_gap < 5:
        print("Evaluación de generalización: excelente")
    elif max_gap < 10:
        print("Evaluación de generalización: buena")
    else:
        print("Evaluación de generalización: con overfitting")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    axs[0].plot(epoch_range, generalization_gaps, marker="o", markersize=4, linewidth=2, color="orange", label="Train Acc - Test Acc")
    axs[0].axhline(5, color="green", linestyle="--", alpha=0.5, label="Umbral de buena generalización (5%)")
    axs[0].axhline(10, color="red", linestyle="--", alpha=0.5, label="Umbral de overfitting (10%)")
    axs[0].fill_between(epoch_range, 0, generalization_gaps, alpha=0.2, color="orange")
    axs[0].set_title("Brecha de Generalización (Overfitting)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Época")
    axs[0].set_ylabel("Brecha (%)")
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    axs[0].set_ylim(0, max(generalization_gaps) + 2)

    subject_ids = [h["subject_id"] for h in all_loso_epochs]
    epochs_to_plot = min(30, max_epochs_all)
    heatmap_data = np.zeros((len(subject_ids), epochs_to_plot))

    for subject_idx, subject_history in enumerate(all_loso_epochs):
        hist = subject_history["history"]
        for epoch_idx in range(epochs_to_plot):
            if epoch_idx < len(hist["epoch"]):
                heatmap_data[subject_idx, epoch_idx] = hist["test_acc"][epoch_idx]

    im = axs[1].imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    axs[1].set_xlabel("Época")
    axs[1].set_ylabel("Sujeto")
    axs[1].set_title("Matriz de Heatmap: Test Acc por Sujeto y Época", fontsize=12, fontweight="bold")
    axs[1].set_xticks(range(0, epochs_to_plot, 5))
    axs[1].set_xticklabels(range(1, epochs_to_plot + 1, 5))
    cbar = plt.colorbar(im, ax=axs[1])
    cbar.set_label("Accuracy (%)", rotation=270, labelpad=20)

    plt.savefig("artifacts/loso_overfitting_analysis.png", dpi=300, bbox_inches="tight")
    print("Gráfico de overfitting guardado: artifacts/loso_overfitting_analysis.png")
    plt.show()
    print(f"Método: LOSO")
    print(f"Total de sujetos: {num_subjects}")
    print(f"Media accuracy: {stats['mean_acc']:.2f}%")
    print(f"Desv. est. accuracy: {stats['std_acc']:.2f}%")
    print(f"IC 95%: [{stats['ci_lower']:.2f}%, {stats['ci_upper']:.2f}%]")
    print(f"Rango accuracy: {stats['min_acc']:.2f}% - {stats['max_acc']:.2f}%")
    print(f"Mejor sujeto 1: {df_results.iloc[0]['subject_id']} ({df_results.iloc[0]['accuracy']:.2f}%)")
    print(f"Mejor sujeto 2: {df_results.iloc[1]['subject_id']} ({df_results.iloc[1]['accuracy']:.2f}%)")
    print(f"Mejor sujeto 3: {df_results.iloc[2]['subject_id']} ({df_results.iloc[2]['accuracy']:.2f}%)")
    print(f"Época de saturación: {saturation_epoch}")
    print(f"Épocas máximas observadas: {max_epochs_all}")
    print(f"Test acc en saturación: {df_convergence.loc[saturation_epoch-1, 'Avg_Test_Acc']:.2f}%")
    print(f"Archivo resultados: {stats['excel_path']}")
    print("Archivo gráfico principal: artifacts/loso_analysis.png")
    print("Archivo gráfico overfitting: artifacts/loso_overfitting_analysis.png")
