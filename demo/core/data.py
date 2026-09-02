"""
data.py

Prepare C-MAPSS TRAIN for a realistic degradation simulation.

Design
------
Original C-MAPSS:
    engine trajectory -> run-to-failure

Our simulation:
    1. Split engines into TRAIN / TEST.
    2. TRAIN:
        - mostly normal observed trajectories
        - a small number of trajectories contain rare abnormal patterns
        - trajectories are NEVER extended beyond their original length
    3. TEST:
        - only an initial fraction of each trajectory is observable
        - the remaining trajectory stays hidden for realtime simulation
    4. Keep original full trajectories untouched for evaluation.

Important semantic separation
-----------------------------
observation cutoff
    != degradation onset
    != failure event
    != rare abnormal event
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from demo.core.fault_injection import (
    DEFAULT_FAULT_METADATA,
    inject_rare_fault,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SplitConfig:
    # Dataset
    data_path: str = "demo/data/train_FD002.txt"
    random_seed: int = 42

    # TRAIN / TEST split
    train_ratio: float = 0.70

    # TRAIN observation boundary
    observation_cycle_min: int = 130
    observation_cycle_max: int = 135

    # Rare abnormal trajectory simulation
    rare_fault_probability: float = 0.20
    rare_fault_max_duration: int = 10
    rare_fault_start_cycle_min: int = 60
    rare_fault_magnitude_min: float = 0.03
    rare_fault_magnitude_max: float = 0.10
    rare_fault_types: tuple[str, ...] = (
        "plateau",
        "drop",
        "drift",
    )

    # TEST observation fraction
    test_fraction_min: float = 0.10
    test_fraction_max: float = 0.30


# ============================================================
# OPERATIONAL SETTINGS & SENSOR DEFINITIONS
# ============================================================

OP_SETTINGS = [
    "op1",
    "op2",
    "op3",
]

ALL_SENSORS = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
    "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
    "Nfdmd", "PCNfRdmd", "W31", "W32",
]

SENSORS = [
    "T24",
    "T30",
    "T50",
    "P30",
    "Ps30",
    "phi",
    "BPR",
    "W31",
    "W32",
]


# ============================================================
# LOAD C-MAPSS
# ============================================================

def load_cmapss_train(config: SplitConfig):
    """
    Load C-MAPSS TRAIN.

    Returns
    -------
    df:
        Original dataframe containing the selected sensors and operational settings.
    engine_data:
        Dictionary {engine_id: ndarray(n_cycles, n_sensors)}
    op_data:
        Dictionary {engine_id: ndarray(n_cycles, 3)}
    """
    columns = ["unit_number", "cycles"] + OP_SETTINGS + ALL_SENSORS

    df = pd.read_csv(
        config.data_path,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    df = df[["unit_number", "cycles"] + OP_SETTINGS + SENSORS].copy()

    engine_data = {}
    op_data = {}

    for engine_id, group in df.groupby("unit_number"):
        group = group.sort_values("cycles")
        engine_data[int(engine_id)] = group[SENSORS].to_numpy(dtype=np.float32)
        op_data[int(engine_id)] = group[OP_SETTINGS].to_numpy(dtype=np.float32)

    return df, engine_data, op_data


# ============================================================
# OBSERVATION CUTOFF
# ============================================================

def sample_observation_cutoff(
    n_cycles: int,
    config: SplitConfig,
    rng: np.random.Generator,
) -> int:
    """
    Sample the observation cutoff for a training engine.
    """
    target_cycle = rng.integers(
        config.observation_cycle_min,
        config.observation_cycle_max + 1,
    )
    cutoff = min(target_cycle, n_cycles)
    return max(1, cutoff)


# ============================================================
# PREPARE TRAIN DATA
# ============================================================

def prepare_train_data(
    train_ids,
    engine_data,
    op_data,
    config: SplitConfig,
    rng: np.random.Generator,
):
    """
    Prepare observed training trajectories.
    """
    train_data = {}
    train_op_data = {}
    train_metadata = {}

    for engine_id in train_ids:
        full = engine_data[engine_id]
        full_op = op_data[engine_id]
        n_cycles = len(full)

        cutoff = sample_observation_cutoff(n_cycles, config, rng)
        observed = full[:cutoff].copy()
        observed_op = full_op[:cutoff].copy()

        inject_fault = rng.random() < config.rare_fault_probability
        if inject_fault:
            observed, fault_metadata = inject_rare_fault(
                trajectory=observed,
                cutoff=len(observed),
                config=config,
                rng=rng,
                sensor_names=SENSORS,
            )
        else:
            fault_metadata = DEFAULT_FAULT_METADATA.copy()

        train_data[engine_id] = observed
        train_op_data[engine_id] = observed_op

        train_metadata[engine_id] = {
            "observed_cycles": len(observed),
            "actual_rul_at_stop": n_cycles - len(observed),
            "total_cycles": n_cycles,
            "is_censored": len(observed) < n_cycles,
            **fault_metadata,
        }

    return train_data, train_op_data, train_metadata


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(
    test_ids,
    engine_data,
    op_data,
    config: SplitConfig,
):
    """
    Prepare initial observations for realtime testing.
    """
    test_rng = np.random.default_rng(config.random_seed + 1)

    test_data = {}
    test_op_data = {}
    test_observation_points = {}

    for engine_id in test_ids:
        full = engine_data[engine_id]
        full_op = op_data[engine_id]
        n_cycles = len(full)

        fraction = test_rng.uniform(
            config.test_fraction_min,
            config.test_fraction_max,
        )

        end = int(np.ceil(n_cycles * fraction))
        end = max(1, min(end, n_cycles - 1))

        test_data[engine_id] = full[:end].copy()
        test_op_data[engine_id] = full_op[:end].copy()
        test_observation_points[engine_id] = end

    return test_data, test_op_data, test_observation_points


# ============================================================
# MAIN SPLIT
# ============================================================

def prepare_cmapss_split(config: SplitConfig | None = None):
    """
    Main entry point.
    """
    if config is None:
        config = SplitConfig()

    # 1. Load original data
    df, engine_data, op_data = load_cmapss_train(config)
    rng = np.random.default_rng(config.random_seed)

    # 2. Engine-level Train / Test Split
    engine_ids = np.array(sorted(engine_data.keys()))
    rng.shuffle(engine_ids)

    n_train = int(len(engine_ids) * config.train_ratio)
    train_ids = sorted(engine_ids[:n_train].tolist())
    test_ids = sorted(engine_ids[n_train:].tolist())

    # 3. Prepare Train
    train_data, train_op_data, train_metadata = prepare_train_data(
        train_ids=train_ids,
        engine_data=engine_data,
        op_data=op_data,
        config=config,
        rng=rng,
    )

    # 4. Prepare Test
    test_data, test_op_data, test_observation_points = prepare_test_data(
        test_ids=test_ids,
        engine_data=engine_data,
        op_data=op_data,
        config=config,
    )

    # 5. Return
    return {
        "df": df,
        "engine_data": engine_data,
        "op_data": op_data,
        "train_data": train_data,
        "train_op_data": train_op_data,
        "train_ids": train_ids,
        "train_metadata": train_metadata,
        "test_data": test_data,
        "test_op_data": test_op_data,
        "test_ids": test_ids,
        "test_observation_points": test_observation_points,
        "sensors": SENSORS,
        "op_settings": OP_SETTINGS,
        "config": config,
    }
