import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phaseI.main import run_phase1
from src.phaseII.main import run_phase2

from demo.core.data import (
    SplitConfig,
    prepare_cmapss_split,
)


config = SplitConfig(
    data_path="demo/data/train_FD002.txt",

    train_ratio=0.70,

    observation_cycle_min=130,
    observation_cycle_max=135,

    rare_fault_probability=0.20,
    rare_fault_max_duration=10,
    rare_fault_start_cycle_min=60,
    rare_fault_magnitude_min=0.03,
    rare_fault_magnitude_max=0.10,
    rare_fault_types=(
        "plateau",
        "drop",
        "drift",
    ),

    test_fraction_min=0.10,
    test_fraction_max=0.30,

    random_seed=42,
)
data = prepare_cmapss_split(config)

from src.preprocess_data.realtime import (
    simulate_realtime,
    RealtimeConfig,
)

realtime = simulate_realtime(
    engine_data=data["engine_data"],
    test_data=data["test_data"],
    observation_points=data["test_observation_points"],
    config=RealtimeConfig(
        update_probability=0.8,
        max_new_cycles=1,
        random_seed=42,
    ),
)


BIN_STRIDE = 10

model, stats, normalized_data, latent_data = run_phase1(
    train_data=data["train_data"],
    n_sensors=len(data["sensors"]),
    window_len=30,
    latent_dim=16,
    bin_stride=BIN_STRIDE,
)

gru_model, history = run_phase2(
    latent_data=latent_data,
    train_metadata=data["train_metadata"],
    latent_dim=16,
)
