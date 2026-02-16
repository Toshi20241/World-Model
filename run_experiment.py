import os
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


@dataclass
class Config:
    env_name: str = "CartPole-v1"
    seed: int = 0
    episodes: int = 150
    max_steps: int = 200
    seq_len: int = 64
    batch_size: int = 256
    hidden_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    device: str = "cpu"
    horizons: tuple = (1, 5, 10, 25, 50)
    train_rollout_steps: int = 25
    results_dir: str = "results"
    use_synthetic_if_gym_fails: bool = True


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot_action(action: int, n_actions: int):
    v = np.zeros(n_actions, dtype=np.float32)
    v[action] = 1.0
    return v


def collect_dataset_cartpole(cfg: Config):
    import gymnasium as gym
    env = gym.make(cfg.env_name)
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    obs_list, act_list, next_obs_list = [], [], []
    for ep in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        for _ in range(cfg.max_steps):
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_list.append(obs.astype(np.float32))
            act_list.append(one_hot_action(action, n_actions))
            next_obs_list.append(next_obs.astype(np.float32))

            obs = next_obs
            if done:
                break

    env.close()
    obs = np.stack(obs_list, axis=0)
    act = np.stack(act_list, axis=0)
    next_obs = np.stack(next_obs_list, axis=0)
    return obs, act, next_obs, obs_dim, n_actions


def collect_dataset_synthetic(cfg: Config):
    n_actions = 2
    obs_dim = 2
    rng = np.random.default_rng(cfg.seed)

    states, actions, next_states = [], [], []
    for _ in range(cfg.episodes):
        x = rng.normal(0.0, 1.0)
        v = rng.normal(0.0, 1.0) * 0.1
        for _ in range(cfg.max_steps):
            a_idx = int(rng.integers(0, 2))
            a = -1.0 if a_idx == 0 else 1.0

            v2 = 0.98 * v + 0.08 * a + 0.02 * np.sin(x)
            x2 = x + v2

            states.append(np.array([x, v], dtype=np.float32))
            actions.append(one_hot_action(a_idx, n_actions))
            next_states.append(np.array([x2, v2], dtype=np.float32))

            x, v = x2, v2

    obs = np.stack(states, axis=0)
    act = np.stack(actions, axis=0)
    next_obs = np.stack(next_states, axis=0)
    return obs, act, next_obs, obs_dim, n_actions


def collect_dataset(cfg: Config):
    try:
        obs, act, next_obs, obs_dim, act_dim = collect_dataset_cartpole(cfg)
        return obs, act, next_obs, obs_dim, act_dim, "CartPole-v1"
    except Exception:
        if not cfg.use_synthetic_if_gym_fails:
            raise
        obs, act, next_obs, obs_dim, act_dim = collect_dataset_synthetic(cfg)
        return obs, act, next_obs, obs_dim, act_dim, "Synthetic2D"


def make_sequences(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, seq_len: int):
    N = obs.shape[0]
    if N <= seq_len + 1:
        raise ValueError("データが少なすぎます。episodes または max_steps を増やしてください。")
    X_obs, X_act, Y_next = [], [], []
    for i in range(0, N - seq_len):
        X_obs.append(obs[i:i + seq_len])
        X_act.append(act[i:i + seq_len])
        Y_next.append(next_obs[i:i + seq_len])
    return np.stack(X_obs, axis=0), np.stack(X_act, axis=0), np.stack(Y_next, axis=0)


def mse_tensor(a: torch.Tensor, b: torch.Tensor):
    d = a - b
    return torch.mean(d * d)


class GRUDynamics(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, obs_dim)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        x = self.inp(x)
        h, _ = self.gru(x)
        y = self.out(h)
        return y


class SimpleSSMDynamics(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
        )
        self.a_raw = nn.Parameter(torch.zeros(hidden))
        self.B = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, obs_dim)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        Bsz, T, _ = obs.shape
        x = torch.cat([obs, act], dim=-1)
        u = self.embed(x)
        a = torch.tanh(self.a_raw)

        s = torch.zeros(Bsz, u.shape[-1], device=u.device)
        preds = []
        for t in range(T):
            s = a * s + self.B(u[:, t])
            preds.append(self.out(s))
        return torch.stack(preds, dim=1)


def count_params(model: nn.Module):
    return int(sum(p.numel() for p in model.parameters()))


def train_one_step(model: nn.Module, loader: DataLoader, cfg: Config):
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    losses = []
    for _ in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for obs_b, act_b, next_b in loader:
            obs_b = obs_b.to(cfg.device)
            act_b = act_b.to(cfg.device)
            next_b = next_b.to(cfg.device)

            pred = model(obs_b, act_b)
            loss = mse_tensor(pred, next_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * obs_b.shape[0]
            n += obs_b.shape[0]
        losses.append(total / max(n, 1))
    return losses


def train_multi_step_gru(gru_model: GRUDynamics, loader: DataLoader, cfg: Config):
    gru_model.to(cfg.device)
    opt = torch.optim.Adam(gru_model.parameters(), lr=cfg.lr)
    losses = []
    K = min(cfg.train_rollout_steps, cfg.seq_len)

    for _ in range(cfg.epochs):
        gru_model.train()
        total, n = 0.0, 0

        for obs_b, act_b, next_b in loader:
            obs_b = obs_b.to(cfg.device)
            act_b = act_b.to(cfg.device)
            next_b = next_b.to(cfg.device)

            cur = obs_b[:, 0]
            loss_acc = 0.0

            for t in range(K):
                a = act_b[:, t]
                pred = gru_model(cur.unsqueeze(1), a.unsqueeze(1))[:, 0]
                true = next_b[:, t]
                loss_acc = loss_acc + mse_tensor(pred, true)
                cur = pred

            loss = loss_acc / float(K)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * obs_b.shape[0]
            n += obs_b.shape[0]

        losses.append(total / max(n, 1))
    return losses


@torch.no_grad()
def rollout_mse(model: nn.Module,
                obs_seq: np.ndarray,
                act_seq: np.ndarray,
                true_next_seq: np.ndarray,
                horizons: tuple,
                device: str):
    model.eval()
    obs_seq_t = torch.tensor(obs_seq, dtype=torch.float32, device=device)
    act_seq_t = torch.tensor(act_seq, dtype=torch.float32, device=device)
    true_next_t = torch.tensor(true_next_seq, dtype=torch.float32, device=device)

    max_h = min(max(horizons), obs_seq_t.shape[1])

    cur = obs_seq_t[:, 0]
    per_step = []
    for t in range(max_h):
        a = act_seq_t[:, t]
        pred = model(cur.unsqueeze(1), a.unsqueeze(1))[:, 0]
        true = true_next_t[:, t]
        per_step.append(float(mse_tensor(pred, true).item()))
        cur = pred

    out = {}
    for h in horizons:
        h2 = min(h, len(per_step))
        out[int(h)] = float(np.mean(per_step[:h2]))
    return out, per_step


@torch.no_grad()
def measure_speed(model: nn.Module, obs_dim: int, act_dim: int, device: str, steps: int = 2000, batch: int = 256):
    model.eval()
    model.to(device)
    obs = torch.randn(batch, 1, obs_dim, device=device)
    act = torch.randn(batch, 1, act_dim, device=device)

    for _ in range(50):
        _ = model(obs, act)

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = model(obs, act)
    t1 = time.perf_counter()

    sec = max(t1 - t0, 1e-9)
    return float((steps * batch) / sec)


def save_line_plot(path, title, xs, ys_list, xlabel, ylabel, labels):
    plt.figure()
    for ys, lab in zip(ys_list, labels):
        plt.plot(xs, ys, marker="o", label=lab)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    cfg = Config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.results_dir, exist_ok=True)
    set_seed(cfg.seed)

    obs, act, next_obs, obs_dim, act_dim, dataset_name = collect_dataset(cfg)
    X_obs, X_act, Y_next = make_sequences(obs, act, next_obs, cfg.seq_len)

    N = X_obs.shape[0]
    idx = int(N * 0.8)

    tr_ds = TensorDataset(
        torch.tensor(X_obs[:idx], dtype=torch.float32),
        torch.tensor(X_act[:idx], dtype=torch.float32),
        torch.tensor(Y_next[:idx], dtype=torch.float32),
    )
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    te_obs = X_obs[idx:]
    te_act = X_act[idx:]
    te_next = Y_next[idx:]
    B_eval = min(512, te_obs.shape[0])
    te_obs = te_obs[:B_eval]
    te_act = te_act[:B_eval]
    te_next = te_next[:B_eval]

    gru = GRUDynamics(obs_dim, act_dim, cfg.hidden_size)
    gru_ms = GRUDynamics(obs_dim, act_dim, cfg.hidden_size)
    ssm = SimpleSSMDynamics(obs_dim, act_dim, cfg.hidden_size)

    loss_gru = train_one_step(gru, tr_loader, cfg)
    loss_gru_ms = train_multi_step_gru(gru_ms, tr_loader, cfg)
    loss_ssm = train_one_step(ssm, tr_loader, cfg)

    xs_epochs = list(range(1, cfg.epochs + 1))
    save_line_plot(
        os.path.join(cfg.results_dir, "train_loss.png"),
        "Training loss",
        xs_epochs,
        [loss_gru, loss_gru_ms, loss_ssm],
        "epoch",
        "MSE",
        ["GRU(one-step)", "GRU(multi-step)", "SSM(simple)"]
    )

    gru_h, _ = rollout_mse(gru, te_obs, te_act, te_next, cfg.horizons, cfg.device)
    gru_ms_h, _ = rollout_mse(gru_ms, te_obs, te_act, te_next, cfg.horizons, cfg.device)
    ssm_h, _ = rollout_mse(ssm, te_obs, te_act, te_next, cfg.horizons, cfg.device)

    xs_h = list(cfg.horizons)
    save_line_plot(
        os.path.join(cfg.results_dir, "mse_vs_horizon.png"),
        "MSE vs horizon",
        xs_h,
        [[gru_h[h] for h in xs_h], [gru_ms_h[h] for h in xs_h], [ssm_h[h] for h in xs_h]],
        "horizon (h)",
        "MSE",
        ["GRU(one-step)", "GRU(multi-step)", "SSM(simple)"]
    )

    models = {
        "GRU(one-step)": (gru, gru_h),
        "GRU(multi-step)": (gru_ms, gru_ms_h),
        "SSM(simple)": (ssm, ssm_h),
    }

    summary = {
        "dataset": dataset_name,
        "episodes": cfg.episodes,
        "max_steps": cfg.max_steps,
        "seq_len": cfg.seq_len,
        "hidden_size": cfg.hidden_size,
        "horizons": list(int(h) for h in cfg.horizons),
        "results": {}
    }

    for name, (m, mse_by_h) in models.items():
        params = count_params(m)
        speed = measure_speed(m, obs_dim, act_dim, cfg.device)
        summary["results"][name] = {
            "params": int(params),
            "speed_steps_per_sec": float(speed),
            "rollout_mse_by_horizon": {str(k): float(v) for k, v in mse_by_h.items()}
        }

    with open(os.path.join(cfg.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # table_numbers.csv（提出用に見やすく）
    lines = []
    lines.append("model,params,steps_per_sec,mse_h1,mse_h10,mse_h50")
    for name in ["GRU(one-step)", "GRU(multi-step)", "SSM(simple)"]:
        r = summary["results"][name]
        mbh = r["rollout_mse_by_horizon"]
        lines.append(",".join([
            name,
            str(r["params"]),
            f"{r['speed_steps_per_sec']:.6g}",
            f"{float(mbh.get('1')):.6g}",
            f"{float(mbh.get('10')):.6g}",
            f"{float(mbh.get('50')):.6g}",
        ]))
    with open(os.path.join(cfg.results_dir, "table_numbers.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Done. Results saved to results/")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
