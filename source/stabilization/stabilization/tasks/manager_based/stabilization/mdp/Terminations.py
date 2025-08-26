# Terminations.py

import math
import torch
from typing import Tuple

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# 관측/보상에서 정의한 함수들을 그대로 재사용
import stabilization.tasks.manager_based.stabilization.mdp as mdp


# ------------------------------------------------------------
# 안정화(stable) 정의: 관측 기반으로만 계산
# ------------------------------------------------------------
def _is_stable_from_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    pos_tol_m: float,
    lin_vel_tol_mps: float,
    ang_vel_tol_radps: float,
    tilt_tol_rad: float,
) -> torch.Tensor:
    """
    관측(ObservationFns)만을 이용해 안정화 여부를 판정합니다.
    - 위치: spawn 기준 오차-norm <= pos_tol_m
    - 속도: 바디 프레임 선속/각속 norm이 임계치 이하
    - 자세: roll/pitch만 포함한 tilt-norm <= tilt_tol_rad (yaw 제외)
    반환: (N,) torch.bool
    """
    # 위치 오차 (spawn 기준)
    pos_err_w = mdp.ObservationFns.position_error_w(env, asset_cfg)  # (N, 3)
    within_pos = torch.linalg.norm(pos_err_w, dim=-1) <= pos_tol_m   # (N,)

    # 선/각속도 (body)
    lin_vel_b = mdp.ObservationFns.lin_vel_body(env, asset_cfg)      # (N, 3)
    ang_vel_b = mdp.ObservationFns.ang_vel_body(env, asset_cfg)      # (N, 3)
    within_lin = torch.linalg.norm(lin_vel_b, dim=-1) <= lin_vel_tol_mps   # (N,)
    within_ang = torch.linalg.norm(ang_vel_b, dim=-1) <= ang_vel_tol_radps # (N,)

    # 기울기(tilt): roll/pitch만 사용 (yaw는 호버링에 독립)
    # roll/pitch가 (N,1)일 가능성이 있어 (N,)으로 맞춤
    roll  = mdp.ObservationFns.roll_current(env, asset_cfg).squeeze(-1)   # (N,)
    pitch = mdp.ObservationFns.pitch_current(env, asset_cfg).squeeze(-1)  # (N,)
    tilt  = torch.sqrt(roll * roll + pitch * pitch)                        # (N,)
    within_tilt = tilt <= tilt_tol_rad                                     # (N,)

    return (within_pos & within_lin & within_ang & within_tilt)            # (N,)


class TerminationFns:
    """
    모든 term은 (N,) 형태의 torch.bool을 반환합니다.
    - 관측 기반의 함수(ObservationFns)만 사용하여 중복 구현을 피함
    """

    @staticmethod
    def flipped_from_obs(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        tilt_threshold_rad: float = math.radians(90.0),
    ) -> torch.Tensor:
        """
        '뒤집힘' 판정(관측 기반):
        roll/pitch로 계산한 tilt = sqrt(roll^2 + pitch^2) > 임계값 → 뒤집힘
        (yaw는 무시)
        """
        roll  = mdp.ObservationFns.roll_current(env, asset_cfg).squeeze(-1)   # (N,)
        pitch = mdp.ObservationFns.pitch_current(env, asset_cfg).squeeze(-1)  # (N,)
        tilt  = torch.sqrt(roll * roll + pitch * pitch)                        # (N,)
        return (tilt > tilt_threshold_rad)                                     # (N,)

    @staticmethod
    def far_from_spawn_from_obs(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        dist_threshold_m: float = 10.0,
    ) -> torch.Tensor:
        """
        스폰 위치로부터의 3D 거리(= |position_error_w|)가 임계값 초과 → 종료
        """
        pos_err_w = mdp.ObservationFns.position_error_w(env, asset_cfg)  # (N, 3)
        dist = torch.linalg.norm(pos_err_w, dim=-1)                      # (N,)
        return (dist > dist_threshold_m)                                  # (N,)

    @staticmethod
    def crashed_from_obs(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        z_min_m: float = 0.02,
    ) -> torch.Tensor:
        """
        충돌 판정(관측 기반):
        절대 고도 z = spawn_z + pos_err_z <= z_min_m 이면 '지면 충돌'로 간주.
        """
        spawn_w   = mdp.ObservationFns.spawn_position_w(env, asset_cfg.name)   # (N, 3)
        pos_err_w = mdp.ObservationFns.position_error_w(env, asset_cfg)        # (N, 3)
        z_abs = (spawn_w + pos_err_w)[..., 2]                                   # (N,)
        return (z_abs <= z_min_m)                                               # (N,)

    @staticmethod
    def stabilized_from_obs(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        pos_tol_m: float = 0.20,
        lin_vel_tol_mps: float = 0.20,
        ang_vel_tol_radps: float = 0.50,
        tilt_tol_rad: float = math.radians(8.0),
    ) -> torch.Tensor:
        """
        '안정화 성공' 조기 종료(성공 종료).
        관측 기반의 안정화 조건을 그대로 사용.
        """
        return _is_stable_from_obs(
            env, asset_cfg, pos_tol_m, lin_vel_tol_mps, ang_vel_tol_radps, tilt_tol_rad
        )  # (N,)

    @staticmethod
    def not_stabilized_timeout_from_obs(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
        timeout_s: float = 15.0,
        pos_tol_m: float = 0.20,
        lin_vel_tol_mps: float = 0.20,
        ang_vel_tol_radps: float = 0.50,
        tilt_tol_rad: float = math.radians(8.0),
    ) -> torch.Tensor:
        """
        '15초 이상 안정화 실패' 타임아웃(Truncation).
        - 경과 스텝 * env.step_dt >= timeout_s 이고
        - 해당 시점에 안정화 조건 미충족 → True
        """
        # 환경에 따라 progress_buf / episode_length_buf / episode_step_count 중 하나를 사용
        if hasattr(env, "episode_length_buf"):
            steps_elapsed = env.episode_length_buf.view(-1)  # (N,)
        elif hasattr(env, "progress_buf"):
            steps_elapsed = env.progress_buf.view(-1)        # (N,)
        else:
            steps_elapsed = env.episode_step_count.view(-1)  # (N,)

        steps_limit = int(timeout_s / env.step_dt + 0.5)
        is_stable = _is_stable_from_obs(
            env, asset_cfg, pos_tol_m, lin_vel_tol_mps, ang_vel_tol_radps, tilt_tol_rad
        )  # (N,)
        return ((steps_elapsed >= steps_limit) & (~is_stable))  # (N,)

    @staticmethod
    def state_nan_or_inf_from_obs(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg(name="Robot"),
    ) -> torch.Tensor:
        """
        수치 불안정 보호(관측 기반):
        관측으로 사용하는 주요 항목들에서 NaN/Inf가 검출되면 종료.
        """
        # roll/pitch/yaw는 (N,1)일 수 있으므로 (N,)로 맞춘 뒤 스택
        roll  = mdp.ObservationFns.roll_current(env, asset_cfg).squeeze(-1)    # (N,)
        pitch = mdp.ObservationFns.pitch_current(env, asset_cfg).squeeze(-1)   # (N,)
        yaw   = mdp.ObservationFns.yaw_current(env, asset_cfg).squeeze(-1)     # (N,)
        pos_err_w = mdp.ObservationFns.position_error_w(env, asset_cfg)        # (N, 3)
        lin_vel_b = mdp.ObservationFns.lin_vel_body(env, asset_cfg)            # (N, 3)
        ang_vel_b = mdp.ObservationFns.ang_vel_body(env, asset_cfg)            # (N, 3)

        scalars = torch.stack([roll, pitch, yaw], dim=-1)                       # (N, 3)
        mats = torch.cat([pos_err_w, lin_vel_b, ang_vel_b], dim=-1)             # (N, 9)

        bad = (
            torch.isnan(scalars).any(dim=-1)
            | torch.isinf(scalars).any(dim=-1)
            | torch.isnan(mats).any(dim=-1)
            | torch.isinf(mats).any(dim=-1)
        )  # (N,)
        return bad


@configclass
class TerminationsCfg:
    """
    Manager 등록용 종료 조건 모음.
    모든 판정은 ObservationFns를 통해 산출된 값(=관측)을 기준으로 수행됩니다.
    """

    # 1) 뒤집힘: tilt(roll,pitch) > 90deg → 실패 종료
    flipped = DoneTerm(
        func=TerminationFns.flipped_from_obs,
        params={"asset_cfg": SceneEntityCfg(name="Robot"), "tilt_threshold_rad": math.radians(90.0)},
        time_out=False,
    )

    # 2) 스폰 기준 10m 이탈 → 실패 종료
    far_from_spawn = DoneTerm(
        func=TerminationFns.far_from_spawn_from_obs,
        params={"asset_cfg": SceneEntityCfg(name="Robot"), "dist_threshold_m": 10.0},
        time_out=False,
    )

    # 3) 지면 충돌(z <= 0.02 m) → 실패 종료
    crashed = DoneTerm(
        func=TerminationFns.crashed_from_obs,
        params={"asset_cfg": SceneEntityCfg(name="Robot"), "z_min_m": 0.02},
        time_out=False,
    )

    # (옵션) 수치 불안정 보호막
    nan_or_inf = DoneTerm(
        func=TerminationFns.state_nan_or_inf_from_obs,
        params={"asset_cfg": SceneEntityCfg(name="Robot")},
        time_out=False,
    )

    # (성공) 안정화 달성 시 조기 종료
    stabilized = DoneTerm(
        func=TerminationFns.stabilized_from_obs,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"),
            "pos_tol_m": 0.20,
            "lin_vel_tol_mps": 0.20,
            "ang_vel_tol_radps": 0.50,
            "tilt_tol_rad": math.radians(8.0),
        },
        time_out=False,
    )

    # (Truncation) 15초 경과에도 안정화 실패
    not_stabilized_timeout = DoneTerm(
        func=TerminationFns.not_stabilized_timeout_from_obs,
        params={
            "asset_cfg": SceneEntityCfg(name="Robot"),
            "timeout_s": 15.0,
            "pos_tol_m": 0.20,
            "lin_vel_tol_mps": 0.20,
            "ang_vel_tol_radps": 0.50,
            "tilt_tol_rad": math.radians(8.0),
        },
        time_out=True,
    )
