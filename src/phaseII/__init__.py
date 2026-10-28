"""
Package phaseII -- HỆ THỐNG DỰ BÁO SUY THOÁI LAI
(Nonlinear Wiener Process with Adaptive Drift & Diffusion)

Kiến trúc mới: scalar latent Z từ Phase I, không dùng W/Codebook.
"""

from src.phaseII.core.schema import (
    SmoothingConfig,
    OnsetConfig,
    WienerConfig,
    FailureThreshold,
    ThresholdScale,
    ThresholdSource,
    HealthState,
    RULStatus,
    FPTSolverStatus,
    SolverDiagnostics,
    FPTResult,
    RULPrediction,
)
from src.phaseII.core.smoother import HealthIndicatorSmoother
from src.phaseII.core.onset_detector import DegradationOnsetDetector
from src.phaseII.core.wiener_process import NonlinearWienerBayesianTracker
from src.phaseII.core.fpt_solver import compute_fleet_failure_threshold, solve_fpt_quantiles
from src.phaseII.core.dashboard import (
    format_cycle_report,
    format_dashboard_table,
    format_dashboard_header_note,
)
from src.phaseII.prognostics_pipeline import PrognosticsPipeline
from src.phaseII.main import run_phase2

__all__ = [
    # Config
    "SmoothingConfig",
    "OnsetConfig",
    "WienerConfig",
    # Enums & value objects
    "FailureThreshold",
    "ThresholdScale",
    "ThresholdSource",
    "HealthState",
    "RULStatus",
    "FPTSolverStatus",
    "SolverDiagnostics",
    "FPTResult",
    # Output
    "RULPrediction",
    # Components
    "HealthIndicatorSmoother",
    "DegradationOnsetDetector",
    "NonlinearWienerBayesianTracker",
    # Solver
    "compute_fleet_failure_threshold",
    "solve_fpt_quantiles",
    # Dashboard
    "format_cycle_report",
    "format_dashboard_table",
    "format_dashboard_header_note",
    # Pipeline
    "PrognosticsPipeline",
    "run_phase2",
]
