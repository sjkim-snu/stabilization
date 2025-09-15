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

from stabilization.tasks.manager_based.stabilization.config import load_parameters  # 추가 (+)
CONFIG = load_parameters()  # 추가 (+)

@dataclass
class CSVLoggerCfg:
    log_dir: Optional[str] = None
    filename: Optional[str] = None
    flush_every_rows: int = 100
    sample_rate: float = 1.0
    device: Optional[str] = None
    policy_dt_s: Optional[float] = None

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

        self._term_names_sorted: List[str] = [  # 추가 (+)
            "abnormal", "ang_vel", "lin_vel", "ori_err", "pos_err", "stabilized", "time_penalty"  # 추가 (+)
        ]  # 추가 (+)

        self._term_weight: Dict[str, float] = {  # 추가 (+)
            "pos_err": float(CONFIG.get("REWARD", {}).get("POS_ERR_WEIGHT", 1.0)),  # 추가 (+)
            "lin_vel": float(CONFIG.get("REWARD", {}).get("LIN_VEL_WEIGHT", 1.0)),  # 추가 (+)
            "ang_vel": float(CONFIG.get("REWARD", {}).get("ANG_VEL_WEIGHT", 1.0)),  # 추가 (+)
            "ori_err": float(CONFIG.get("REWARD", {}).get("ORI_ERR_WEIGHT", 1.0)),  # 추가 (+)
            "time_penalty": float(CONFIG.get("REWARD", {}).get("TIME_PENALTY_WEIGHT", 1.0)),  # 추가 (+)
            "stabilized": float(CONFIG.get("REWARD", {}).get("STABILIZED_BONUS_WEIGHT", 1.0)),  # 추가 (+)
            "abnormal": float(CONFIG.get("REWARD", {}).get("ABNORMAL_PENALTY_WEIGHT", 1.0)),  # 추가 (+)
        }  # 추가 (+)

        self._csv_file = open(self.filepath, "w", newline="", encoding="utf-8")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._header_written = False
        self._rows_since_flush = 0

        self._rew_total_sum: Optional[torch.Tensor] = None  # 추가 (+)
        self._term_wsum: Dict[str, torch.Tensor] = {}  # 추가 (+)
        self._act_sum: Optional[torch.Tensor] = None  # 추가 (+)

    def _ensure_device_and_alloc(self, ref_tensor: torch.Tensor):
        if self._torch_device is None:
            self._torch_device = ref_tensor.device
        if not self._core_allocated:
            dev = self._torch_device
            self._ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
            self._ep_idx = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
            self._rew_total_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)  # 추가 (+)
            for name in self._term_names_sorted:  # 추가 (+)
                self._term_wsum[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)  # 추가 (+)
            self._act_sum = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=dev)  # 추가 (+)
            self._core_allocated = True

    def _ensure_terms(self, term_names: List[str]):
        dev = self._torch_device if self._torch_device is not None else torch.device("cpu")
        for name in term_names:
            if name not in self._term_wsum:
                self._term_wsum[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=dev)

    def _build_writer_if_needed(self):
        if self._csv_writer is not None:
            return
        header = [  # 추가 (+)
            "timestamp_kst",  # 추가 (+)
            "env_id",  # 추가 (+)
            "episode_idx",  # 추가 (+)
            "episode_length",  # 추가 (+)
            "episode_length_s",  # 추가 (+)
            "done_reason",  # 추가 (+)
            "rew_abnormal_sum_w",
            "rew_ang_vel_sum_w",
            "rew_lin_vel_sum_w",
            "rew_ori_err_sum_w",
            "rew_pos_err_sum_w",
            "rew_stabilized_sum_w",
            "rew_time_penalty_sum_w",
            "rew_total_sum",
            "act_mean_0",  # 추가 (+)
            "act_mean_1",  # 추가 (+)
            "act_mean_2",  # 추가 (+)
            "act_mean_3",  # 추가 (+)
        ]  # 추가 (+)
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=header)
        self._csv_writer.writeheader()
        self._header_written = True

    def _ts_kst(self) -> str:
        return datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S") if _KST else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _done_reason(self, term_mgr, i: int) -> str:  # 추가 (+)
        reason = "unknown"  # 추가 (+)
        try:  # 추가 (+)
            if term_mgr is None:  # 추가 (+)
                return reason  # 추가 (+)
            if hasattr(term_mgr, "time_outs") and bool(term_mgr.time_outs[i]):  # 추가 (+)
                return "time_out"  # 추가 (+)
            if hasattr(term_mgr, "active_terms"):  # 추가 (+)
                for name in term_mgr.active_terms:  # 추가 (+)
                    try:  # 추가 (+)
                        term = term_mgr.get_term(name)  # 추가 (+)
                        if term is not None and bool(term[i]):  # 추가 (+)
                            return name  # 추가 (+)
                    except Exception:  # 추가 (+)
                        continue  # 추가 (+)
        except Exception:  # 추가 (+)
            pass  # 추가 (+)
        return reason  # 추가 (+)

    @torch.no_grad()
    def log_step(  # 추가 (+)
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rew_terms_step: Optional[Dict[str, torch.Tensor]] = None,
        term_mgr=None,
        actions: Optional[torch.Tensor] = None,
    ):
        N = rewards.shape[0]
        if N != self.num_envs:
            raise ValueError("num_envs mismatch")

        self._ensure_device_and_alloc(rewards)

        try:
            dones = dones.reshape(-1).to(dtype=torch.bool, device=self._torch_device)
        except Exception:
            dones = torch.as_tensor(dones, dtype=torch.bool, device=self._torch_device).reshape(-1)

        self._ep_len += 1
        self._rew_total_sum += rewards.to(device=self._torch_device, dtype=torch.float32).reshape(-1)

        if rew_terms_step:
            self._ensure_terms(list(rew_terms_step.keys()))
            for name, val in rew_terms_step.items():
                w = float(self._term_weight.get(name, 1.0))
                self._term_wsum[name] += (val * w * float(self.cfg.policy_dt_s))

        if actions is not None:  # 추가 (+)
            A = actions.shape[-1]  # 추가 (+)
            use = min(A, 4)  # 추가 (+)
            if use > 0:  # 추가 (+)
                self._act_sum[:, :use] += actions[:, :use].to(device=self._torch_device, dtype=torch.float32)  # 추가 (+)

        done_ids = torch.nonzero(dones).squeeze(-1)
        if done_ids.numel() == 0:
            return

        if not self._header_written:
            self._build_writer_if_needed()

        for i in done_ids.tolist():
            if self.cfg.sample_rate < 1.0 and random.random() > self.cfg.sample_rate:
                self._reset_one_env(i)
                continue

            ep_len = int(self._ep_len[i].item())  # 추가 (+)
            ep_len_f = f"{float(ep_len):.6f}"  # 추가 (+)
            ep_len_s = f"{(float(ep_len) * self.cfg.policy_dt_s):.6f}" if self.cfg.policy_dt_s is not None else ""  # 추가 (+)

            row: Dict[str, str] = {}
            row["timestamp_kst"] = self._ts_kst()  # 추가 (+)
            row["env_id"] = i  # 추가 (+)
            row["episode_idx"] = int(self._ep_idx[i].item())  # 추가 (+)
            row["episode_length"] = ep_len_f  # 추가 (+)
            row["episode_length_s"] = ep_len_s  # 추가 (+)
            row["done_reason"] = self._done_reason(term_mgr, i)  # 추가 (+)

            sw_abn = float(self._term_wsum.get('abnormal', torch.zeros_like(self._rew_total_sum))[i].item())  # 추가 (+)
            sw_ang = float(self._term_wsum.get('ang_vel', torch.zeros_like(self._rew_total_sum))[i].item())  # 추가 (+)
            sw_lin = float(self._term_wsum.get('lin_vel', torch.zeros_like(self._rew_total_sum))[i].item())  # 추가 (+)
            sw_ori = float(self._term_wsum.get('ori_err', torch.zeros_like(self._rew_total_sum))[i].item())  # 추가 (+)
            sw_pos = float(self._term_wsum.get('pos_err', torch.zeros_like(self._rew_total_sum))[i].item())  # 추가 (+)
            sw_sta = float(self._term_wsum.get('stabilized', torch.zeros_like(self._rew_total_sum))[i].item())  # 추가 (+)
            sw_time = float(self._term_wsum.get('time_penalty', torch.zeros_like(self._rew_total_sum))[i].item())  # 추가 (+)

            row["rew_abnormal_sum_w"] = f"{sw_abn:.6f}"
            row["rew_ang_vel_sum_w"] = f"{sw_ang:.6f}"
            row["rew_lin_vel_sum_w"] = f"{sw_lin:.6f}"
            row["rew_ori_err_sum_w"] = f"{sw_ori:.6f}"
            row["rew_pos_err_sum_w"] = f"{sw_pos:.6f}"
            row["rew_stabilized_sum_w"] = f"{sw_sta:.6f}"
            row["rew_time_penalty_sum_w"] = f"{sw_time:.6f}"

            row["rew_total_sum"] = f"{float(self._rew_total_sum[i].item()):.6f}"

            if self._act_sum is not None and ep_len > 0:  # 추가 (+)
                means = (self._act_sum[i] / float(ep_len)).detach().to("cpu").tolist()  # 추가 (+)
                means = (means + [0.0, 0.0, 0.0, 0.0])[:4]  # 추가 (+)
                row["act_mean_0"] = f"{float(means[0]):.6f}"  # 추가 (+)
                row["act_mean_1"] = f"{float(means[1]):.6f}"  # 추가 (+)
                row["act_mean_2"] = f"{float(means[2]):.6f}"  # 추가 (+)
                row["act_mean_3"] = f"{float(means[3]):.6f}"  # 추가 (+)

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
        for t in list(self._term_wsum.keys()):
            self._term_wsum[t][i] = 0.0
        self._rew_total_sum[i] = 0.0  # 추가 (+)
        if self._act_sum is not None:  # 추가 (+)
            self._act_sum[i, :] = 0.0  # 추가 (+)

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
