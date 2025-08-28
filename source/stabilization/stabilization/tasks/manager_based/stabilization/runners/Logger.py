from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import csv, os, torch, random

try:
    from zoneinfo import ZoneInfo
    _KST = ZoneInfo("Asia/Seoul")
except Exception:
    _KST = None

@dataclass
class CSVLoggerCfg:
    log_dir: Optional[str] = None
    filename: Optional[str] = None
    flush_every_rows: int = 100
    sample_rate: float = 1.0
    device: Optional[str] = None           # "cuda:0" or "cpu"
    policy_dt_s: Optional[float] = None    # physics_dt * decimation

class EpisodeCSVLogger:
    def __init__(self, num_envs: int, cfg: CSVLoggerCfg = CSVLoggerCfg()):
        self.num_envs = num_envs
        self.cfg = cfg

        if self.cfg.log_dir is None:
            self.cfg.log_dir = str((Path(__file__).resolve().parents[1] / "log").resolve())
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)

        if self.cfg.filename is None:
            stamp = datetime.now(_KST).strftime("%Y%m%d_%H%M%S") if _KST else datetime.now().strftime("%Y%m%d_%H%M%S")
            self.cfg.filename = f"{stamp}.csv"

        self.filepath = str((Path(self.cfg.log_dir) / self.cfg.filename).resolve())

        self._torch_device: Optional[torch.device] = torch.device(self.cfg.device) if self.cfg.device else None
        self._core_allocated = False

        self._ep_len: Optional[torch.Tensor] = None
        self._ep_idx: Optional[torch.Tensor] = None
        self._total_sum: Optional[torch.Tensor] = None
        self._total_sumsq: Optional[torch.Tensor] = None

        self._term_sum: Dict[str, torch.Tensor] = {}
        self._term_sumsq: Dict[str, torch.Tensor] = {}
        self._term_names_sorted: Optional[List[str]] = None

        self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._header_written = False
        self._rows_since_flush = 0

    def _ensure_device_and_alloc(self, ref_tensor: torch.Tensor):
        if self._torch_device is None:
            self._torch_device = ref_tensor.device
        if not self._core_allocated:
            dev = self._torch_device
            self._ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
            self._ep_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
            self._total_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)
            self._total_sumsq = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)
            self._core_allocated = True

    def _ensure_terms(self, term_names: List[str]):
        dev = self._torch_device if self._torch_device is not None else torch.device("cpu")
        for name in term_names:
            if name not in self._term_sum:
                self._term_sum[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)
                self._term_sumsq[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)

    @staticmethod
    def _mean_var(s: float, s2: float, n: int):
        if n <= 0:
            return 0.0, 0.0
        mean = s / n
        var = s2 / n - mean * mean
        return mean, max(var, 0.0)

    def _build_writer_if_needed(self):
        if self._csv_writer is not None:
            return
        if self._term_names_sorted is None:
            self._term_names_sorted = sorted(self._term_sum.keys())

        header = [
            "timestamp_kst",
            "env_id",
            "episode_idx",
            "episode_length",          # steps
            "episode_length_s",        # seconds (policy_dt_s가 설정된 경우)
            "done_reason",
        ]
        for t in self._term_names_sorted:
            header += [f"rew_{t}_sum", f"rew_{t}_mean", f"rew_{t}_var"]
        header += ["reward_total_sum", "reward_total_mean", "reward_total_var"]

        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=header)
        self._csv_writer.writeheader()
        self._header_written = True

    def _ts_kst(self) -> str:
        return datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S") if _KST else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @torch.no_grad()
    def log_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rew_terms_step: Optional[Dict[str, torch.Tensor]] = None,
        term_mgr=None,
    ):
        N = rewards.shape[0]
        if N != self.num_envs:
            raise ValueError("num_envs mismatch")

        self._ensure_device_and_alloc(rewards)

        self._ep_len += 1
        self._total_sum += rewards
        self._total_sumsq += rewards * rewards

        if rew_terms_step:
            self._ensure_terms(list(rew_terms_step.keys()))
            for name, val in rew_terms_step.items():
                self._term_sum[name] += val
                self._term_sumsq[name] += val * val

        done_ids = torch.nonzero(dones).squeeze(-1)
        if done_ids.numel() == 0:
            return

        if not self._header_written:
            self._build_writer_if_needed()

        for i in done_ids.tolist():
            if self.cfg.sample_rate < 1.0 and random.random() > self.cfg.sample_rate:
                self._reset_one_env(i)
                continue

            ep_len = int(self._ep_len[i].item())
            ep_len_s = f"{(ep_len * self.cfg.policy_dt_s):.6f}" if self.cfg.policy_dt_s is not None else ""

            reason = "unknown"
            if term_mgr is not None:
                try:
                    if hasattr(term_mgr, "time_outs") and bool(term_mgr.time_outs[i]):
                        reason = "time_out"
                    elif hasattr(term_mgr, "active_terms"):
                        for name in term_mgr.active_terms:
                            if bool(term_mgr.get_term(name)[i]):
                                reason = name
                                break
                except Exception:
                    pass

            row = {
                "timestamp_kst": self._ts_kst(),
                "env_id": i,
                "episode_idx": int(self._ep_idx[i].item()),
                "episode_length": ep_len,
                "episode_length_s": ep_len_s,
                "done_reason": reason,
            }

            if self._term_names_sorted is None:
                self._term_names_sorted = sorted(self._term_sum.keys())
            for t in self._term_names_sorted:
                s  = float(self._term_sum[t][i].item())
                s2 = float(self._term_sumsq[t][i].item())
                m, v = self._mean_var(s, s2, ep_len)
                row[f"rew_{t}_sum"]  = f"{s:.6f}"
                row[f"rew_{t}_mean"] = f"{m:.6f}"
                row[f"rew_{t}_var"]  = f"{v:.6f}"

            s_tot  = float(self._total_sum[i].item())
            s2_tot = float(self._total_sumsq[i].item())
            m_tot, v_tot = self._mean_var(s_tot, s2_tot, ep_len)
            row["reward_total_sum"]  = f"{s_tot:.6f}"
            row["reward_total_mean"] = f"{m_tot:.6f}"
            row["reward_total_var"]  = f"{v_tot:.6f}"

            self._csv_writer.writerow(row)
            self._rows_since_flush += 1
            if self._rows_since_flush >= self.cfg.flush_every_rows:
                try:
                    self._csv_file.flush()
                    os.fsync(self._csv_file.fileno())
                except Exception:
                    pass
                self._rows_since_flush = 0

            self._reset_one_env(i)

    def _reset_one_env(self, i: int):
        self._ep_idx[i] += 1
        self._ep_len[i] = 0
        self._total_sum[i] = 0.0
        self._total_sumsq[i] = 0.0
        for t in self._term_sum.keys():
            self._term_sum[t][i] = 0.0
            self._term_sumsq[t][i] = 0.0

    def close(self):
        try:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
        except Exception:
            pass
        try:
            self._csv_file.close()
        except Exception:
            pass
