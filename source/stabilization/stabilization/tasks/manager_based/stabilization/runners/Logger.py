from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import csv, os, torch, random

# use KST timezone if available
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
    sample_rate: float = 1.0               # if 1.0, all episodes are logged
    device: Optional[str] = None           # "cuda:0" or "cpu"
    policy_dt_s: Optional[float] = None    # physics_dt * decimation

class EpisodeCSVLogger:
    
    def __init__(self, num_envs: int, cfg: CSVLoggerCfg = CSVLoggerCfg()):
        
        """
        Initialize the CSV logger for episode data.
        Args:
            num_envs (int): Number of entities.
            cfg (CSVLoggerCfg): Configuration for the logger.
        """
        self.num_envs = num_envs
        self.cfg = cfg

        # Create log directory if it doesn't exist
        if self.cfg.log_dir is None:
            self.cfg.log_dir = str((Path(__file__).resolve().parents[1] / "log").resolve())
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp if not provided
        if self.cfg.filename is None:
            stamp = datetime.now(_KST).strftime("%Y%m%d_%H%M%S") if _KST else datetime.now().strftime("%Y%m%d_%H%M%S")
            self.cfg.filename = f"{stamp}.csv"

        # Set file path and name
        self.filepath = str((Path(self.cfg.log_dir) / self.cfg.filename).resolve())
        
        # If device is specified, use it; otherwise, first tensor's device will be used
        self._torch_device: Optional[torch.device] = torch.device(self.cfg.device) if self.cfg.device else None
        
        # Initialize internal state
        self._core_allocated = False
        self._ep_len: Optional[torch.Tensor] = None
        self._ep_idx: Optional[torch.Tensor] = None

        # Reward term accumulators (per-env)
        self._term_sum: Dict[str, torch.Tensor] = {}
        self._term_names_sorted: Optional[List[str]] = None

        # Action logging: episode-wise mean of action components (fixed 4 dims as requested)
        self._act_sum: Optional[torch.Tensor] = None  # shape: [num_envs, 4]

        # Open CSV file for writing
        self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._header_written = False
        self._rows_since_flush = 0
    
    # Ensure tensors are allocated on the correct device
    def _ensure_device_and_alloc(self, ref_tensor: torch.Tensor):
        if self._torch_device is None:
            self._torch_device = ref_tensor.device
        if not self._core_allocated:
            dev = self._torch_device
            self._ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
            self._ep_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
            # action accumulators (4-dim mean vector)
            self._act_sum = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=dev)
            self._core_allocated = True

    # Ensure reward terms are initialized
    def _ensure_terms(self, term_names: List[str]):
        dev = self._torch_device if self._torch_device is not None else torch.device("cpu")
        for name in term_names:
            if name not in self._term_sum:
                self._term_sum[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)

    def _build_writer_if_needed(self):
        if self._csv_writer is not None:
            return
        if self._term_names_sorted is None:
            self._term_names_sorted = sorted(self._term_sum.keys())

        header = [
            "timestamp_kst",
            "env_id",
            "episode_idx",
            "episode_length",          # float steps
            "episode_length_s",        # float seconds (policy_dt_s가 설정된 경우)
            "done_reason",
        ]
        # reward terms: sum only
        for t in self._term_names_sorted:
            header += [f"rew_{t}_sum"]

        # action mean (episode-wise, 4-dim)
        header += [f"act_mean_{k}" for k in range(4)]

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
        actions: Optional[torch.Tensor] = None,  # [N, A], A can be >=,<,= 4
    ):
        N = rewards.shape[0]
        if N != self.num_envs:
            raise ValueError("num_envs mismatch")

        self._ensure_device_and_alloc(rewards)

        # episode step accumulations
        self._ep_len += 1

        # reward terms: accumulate only sums
        if rew_terms_step:
            self._ensure_terms(list(rew_terms_step.keys()))
            for name, val in rew_terms_step.items():
                self._term_sum[name] += val

        # actions: accumulate for episode-wise mean (fixed 4 dims)
        if actions is not None:
            A = actions.shape[-1]
            # add first min(A,4) components
            use = min(A, 4)
            if use > 0:
                self._act_sum[:, :use] += actions[:, :use]
            # if action dim < 4, the remaining components stay accumulated as 0

        # write rows for done envs
        done_ids = torch.nonzero(dones).squeeze(-1)
        if done_ids.numel() == 0:
            return

        # build header lazily when we know term names
        if not self._header_written:
            self._build_writer_if_needed()

        # if sample_rate < 1.0, randomly skip some episodes
        for i in done_ids.tolist():
            if self.cfg.sample_rate < 1.0 and random.random() > self.cfg.sample_rate:
                self._reset_one_env(i)
                continue

            ep_len = int(self._ep_len[i].item())
            # as requested: both fields are float strings
            ep_len_f = f"{float(ep_len):.6f}"
            ep_len_s = f"{(float(ep_len) * self.cfg.policy_dt_s):.6f}" if self.cfg.policy_dt_s is not None else ""

            # determine done reason
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
                "episode_length": ep_len_f,
                "episode_length_s": ep_len_s,
                "done_reason": reason,
            }

            # reward term fields (sum only)
            if self._term_names_sorted is None:
                self._term_names_sorted = sorted(self._term_sum.keys())
            for t in self._term_names_sorted:
                s = float(self._term_sum[t][i].item())
                row[f"rew_{t}_sum"] = f"{s:.6f}"

            # action mean across steps (4 components as float)
            if self._act_sum is not None and ep_len > 0:
                means = (self._act_sum[i] / float(ep_len)).detach().to("cpu").tolist()
                # ensure 4 outputs
                means = (means + [0.0, 0.0, 0.0, 0.0])[:4]
                for k in range(4):
                    row[f"act_mean_{k}"] = f"{float(means[k]):.6f}"

            # write row
            self._csv_writer.writerow(row)
            self._rows_since_flush += 1
            if self._rows_since_flush >= self.cfg.flush_every_rows:
                try:
                    self._csv_file.flush()
                    os.fsync(self._csv_file.fileno())
                except Exception:
                    pass
                self._rows_since_flush = 0

            # reset episode accumulators for this env
            self._reset_one_env(i)

    def _reset_one_env(self, i: int):
        # increase episode index and clear accumulators
        self._ep_idx[i] += 1
        self._ep_len[i] = 0
        for t in self._term_sum.keys():
            self._term_sum[t][i] = 0.0
        if self._act_sum is not None:
            self._act_sum[i, :] = 0.0

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
