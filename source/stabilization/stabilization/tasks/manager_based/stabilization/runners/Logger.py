from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import csv, os, torch

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
            now = datetime.now(_KST) if _KST else datetime.now()
            self.cfg.filename = f"{now.strftime('%Y%m%d_%H%M%S')}.csv"

        self.filepath = str((Path(self.cfg.log_dir) / self.cfg.filename).resolve())
        self._torch_device: Optional[torch.device] = torch.device(self.cfg.device) if self.cfg.device else None

        self._core_allocated = False
        self._ep_len: Optional[torch.Tensor] = None
        self._ep_idx: Optional[torch.Tensor] = None
        self._rew_total_sum: Optional[torch.Tensor] = None
        self._term_wsum: Dict[str, torch.Tensor] = {}
        self._act_sum: Optional[torch.Tensor] = None
        self._act_dim: Optional[int] = None

        self._term_weight: Dict[str, float] = {  # 추가 (+)
            "pos_err": float(CONFIG.get("REWARD", {}).get("POS_ERR_WEIGHT", 1.0)),  # 추가 (+)
            "lin_vel": float(CONFIG.get("REWARD", {}).get("LIN_VEL_WEIGHT", 1.0)),  # 추가 (+)
            "ang_vel": float(CONFIG.get("REWARD", {}).get("ANG_VEL_WEIGHT", 1.0)),  # 추가 (+)
            "ori_err": float(CONFIG.get("REWARD", {}).get("ORI_ERR_WEIGHT", 1.0)),  # 추가 (+)
            "time_penalty": float(CONFIG.get("REWARD", {}).get("TIME_PENALTY_WEIGHT", 1.0)),  # 추가 (+)
            "stabilized": float(CONFIG.get("REWARD", {}).get("STABILIZED_BONUS_WEIGHT", 1.0)),  # 추가 (+)
            "abnormal": float(CONFIG.get("REWARD", {}).get("ABNORMAL_PENALTY_WEIGHT", 1.0)),  # 추가 (+)
        }  # 추가 (+)

        self._term_alias: Dict[str, str] = {  # 추가 (+)
            "pos_err_w": "pos_err",  # 추가 (+)
            "lin_vel_w": "lin_vel",  # 추가 (+)
            "ang_vel_b": "ang_vel",  # 추가 (+)
            "orientation": "ori_err",  # 추가 (+)
            "time_penalty": "time_penalty",  # 추가 (+)
            "stabilized_bonus": "stabilized",  # 추가 (+)
            "abnormal_penalty": "abnormal",  # 추가 (+)
        }  # 추가 (+)
        for _k, _v in list(self._term_alias.items()):  # 추가 (+)
            if _k not in self._term_weight:  # 추가 (+)
                self._term_weight[_k] = float(self._term_weight.get(_v, 1.0))  # 추가 (+)

        self._csv_file = open(self.filepath, "w", newline="")
        self._csv_writer = None
        self._rows_since_flush = 0

    def _allocate_core(self, actions: Optional[torch.Tensor] = None):
        device = self._torch_device if self._torch_device is not None else torch.device("cpu")
        self._ep_len = torch.zeros(self.num_envs, dtype=torch.int32, device=device)
        self._ep_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self._rew_total_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        need_terms = ["pos_err", "ang_vel", "lin_vel", "ori_err", "stabilized", "abnormal", "time_penalty"]  # 추가 (+)
        for name in set(need_terms + list(self._term_alias.keys())):  # 추가 (+)
            self._term_wsum[name] = torch.zeros(self.num_envs, dtype=torch.float32, device=device)  # 추가 (+)
        if actions is not None:
            self._act_dim = int(actions.shape[-1])
            self._act_sum = torch.zeros(self.num_envs, self._act_dim, dtype=torch.float32, device=device)
        self._core_allocated = True

    def _ensure_terms(self, keys: List[str]):
        device = self._torch_device if self._torch_device is not None else torch.device("cpu")
        for k in keys:
            base = self._term_alias.get(k, k)
            if base not in self._term_wsum:
                self._term_wsum[base] = torch.zeros(self.num_envs, dtype=torch.float32, device=device)  # 추가 (+)
            if k != base and k not in self._term_wsum:
                self._term_wsum[k] = torch.zeros(self.num_envs, dtype=torch.float32, device=device)  # 추가 (+)

    def _write_header(self):
        header = [
            "timestamp",
            "env_id",
            "episode_idx",
            "episode_length",
            "episode_length_s",
            "done_reason",
            "rew_pos_err_sum_w",
            "rew_ang_vel_sum_w",
            "rew_lin_vel_sum_w",
            "rew_ori_err_sum_w",
            "rew_stabilized_sum_w",
            "rew_abnormal_sum_w",
            "rew_time_penalty_sum_w",
            "rew_total_sum",
            "act_mean_0",
            "act_mean_1",
            "act_mean_2",
            "act_mean_3",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=header)
        self._csv_writer.writeheader()

    def _extract_done_reason(self, term_mgr, env_i: int) -> str:  # 추가 (+)
        try:  # 추가 (+)
            if term_mgr is None:  # 추가 (+)
                return ""  # 추가 (+)
            # 1) time_out 우선 확인  # 추가 (+)
            if hasattr(term_mgr, "time_outs"):  # 추가 (+)
                to = term_mgr.time_outs  # 추가 (+)
                if isinstance(to, torch.Tensor) and env_i < to.shape[0] and bool(to[env_i].item()):  # 추가 (+)
                    return "time_out"  # 추가 (+)
            # 2) 활성화된 term을 직접 조회 (가장 신뢰도 높음)  # 추가 (+)
            if hasattr(term_mgr, "get_active_iterable_terms"):  # 추가 (+)
                seq = term_mgr.get_active_iterable_terms(env_i)  # [(name, [val]), ...]  # 추가 (+)
                reasons = [name for name, vals in seq if any(float(v) > 0.5 for v in vals)]  # 추가 (+)
                if reasons:  # 추가 (+)
                    return reasons[0]  # 추가 (+)
            # 3) fallback: active_terms 순회하며 get_term(name)[env_i]가 True인 항목 찾기  # 추가 (+)
            names = getattr(term_mgr, "active_terms", None)  # 추가 (+)
            if isinstance(names, (list, tuple)):  # 추가 (+)
                for name in names:  # 추가 (+)
                    try:  # 추가 (+)
                        t = term_mgr.get_term(name)  # 추가 (+)
                        if isinstance(t, torch.Tensor) and env_i < t.shape[0] and bool(t[env_i].item()):  # 추가 (+)
                            return name  # 추가 (+)
                    except Exception:  # 추가 (+)
                        continue  # 추가 (+)
            # 4) 최후 fallback: terminated 플래그만 True인 경우  # 추가 (+)
            if hasattr(term_mgr, "terminated"):  # 추가 (+)
                term = term_mgr.terminated  # 추가 (+)
                if isinstance(term, torch.Tensor) and env_i < term.shape[0] and bool(term[env_i].item()):  # 추가 (+)
                    return "terminated"  # 추가 (+)
        except Exception:  # 추가 (+)
            pass  # 추가 (+)
        return ""  # 추가 (+)


    def log_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rew_terms_step: Optional[Dict[str, torch.Tensor]] = None,
        term_mgr=None,
        actions: Optional[torch.Tensor] = None,
    ):
        if not self._core_allocated:
            self._allocate_core(actions)

        self._ep_len += 1
        self._rew_total_sum += rewards.to(self._rew_total_sum.device)

        if actions is not None and self._act_sum is not None:
            self._act_sum += actions.to(self._act_sum.device)

        if rew_terms_step:
            keys = list(rew_terms_step.keys())  # 추가 (+)
            self._ensure_terms(keys)  # 추가 (+)
            for name, val in rew_terms_step.items():
                base = self._term_alias.get(name, name)  # 추가 (+)
                w = float(self._term_weight.get(name, self._term_weight.get(base, 1.0)))  # 추가 (+)
                dt = float(self.cfg.policy_dt_s) if self.cfg.policy_dt_s is not None else 1.0  # 추가 (+)
                device = self._term_wsum[base].device  # 추가 (+)
                self._term_wsum[base] += (val.to(device) * w * dt)  # 추가 (+)
                if name != base:
                    self._term_wsum[name] += (val.to(device) * w * dt)  # 추가 (+)

        done_idx = torch.nonzero(dones, as_tuple=False).flatten().tolist()
        if not done_idx:
            return

        now = datetime.now(_KST) if _KST else datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        if self._csv_writer is None:
            self._write_header()

        for i in done_idx:
            row = {}
            row["timestamp"] = ts
            row["env_id"] = int(i)
            row["episode_idx"] = int(self._ep_idx[i].item())
            ep_len_i = int(self._ep_len[i].item())
            row["episode_length"] = ep_len_i
            dt = float(self.cfg.policy_dt_s) if self.cfg.policy_dt_s is not None else 1.0  # 추가 (+)
            row["episode_length_s"] = f"{ep_len_i * dt:.6f}"  # 추가 (+)
            row["done_reason"] = self._extract_done_reason(term_mgr, i)  # 추가 (+)

            def _get_sum(name: str) -> float:
                t = self._term_wsum.get(name, None)
                if t is None:
                    return 0.0
                return float(t[i].item())

            row["rew_pos_err_sum_w"] = f"{_get_sum('pos_err'):.6f}"
            row["rew_ang_vel_sum_w"] = f"{_get_sum('ang_vel'):.6f}"
            row["rew_lin_vel_sum_w"] = f"{_get_sum('lin_vel'):.6f}"
            row["rew_ori_err_sum_w"] = f"{_get_sum('ori_err'):.6f}"
            row["rew_stabilized_sum_w"] = f"{_get_sum('stabilized'):.6f}"
            row["rew_abnormal_sum_w"] = f"{_get_sum('abnormal'):.6f}"
            row["rew_time_penalty_sum_w"] = f"{_get_sum('time_penalty'):.6f}"
            row["rew_total_sum"] = f"{float(self._rew_total_sum[i].item()):.6f}"

            if self._act_sum is not None and self._act_dim is not None and ep_len_i > 0:
                means = (self._act_sum[i] / float(ep_len_i)).tolist()
            else:
                means = []
            for j in range(4):
                val = means[j] if j < len(means) else None
                row[f"act_mean_{j}"] = (f"{float(val):.6f}" if val is not None else "")

            self._csv_writer.writerow(row)
            self._rows_since_flush += 1

            self._ep_idx[i] += 1
            self._ep_len[i] = 0
            self._rew_total_sum[i] = 0.0
            for k in list(self._term_wsum.keys()):
                self._term_wsum[k][i] = 0.0
            if self._act_sum is not None:
                self._act_sum[i, :] = 0.0

        if self.cfg.flush_every_rows is not None and self._rows_since_flush >= int(self.cfg.flush_every_rows):
            try:
                self._csv_file.flush()
                os.fsync(self._csv_file.fileno())
            except Exception:
                pass
            self._rows_since_flush = 0

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
