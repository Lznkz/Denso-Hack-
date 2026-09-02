"""
prepare_data.py -- Chuẩn bị dữ liệu Phase II cho FD002.

Input:
    latent_data      : dict[engine_id -> z_seq (n_bins, z_dim)], từ
                        train.extract_latents_for_engine (Phase I đã freeze).
    true_failure_bin : dict[engine_id -> int], CHỈ chứa các engine thuộc
                        nhóm "giữ nguyên, không cắt" -- bin cuối cùng
                        trong latent_seq của chính engine đó, tương ứng
                        đúng cycle hỏng THẬT (dòng cuối chuỗi gốc FD002,
                        vì train_FD002.txt vốn dĩ là run-to-failure).
                        Engine KHÔNG có mặt trong dict này => censored.

Nhiệm vụ:
    1. Build EngineRecord: event_bin lấy TRỰC TIẾP từ true_failure_bin
       nếu có -- KHÔNG suy đoán qua z_drift/threshold (đây là nhãn THẬT,
       biết chắc, khác hẳn cách tiếp cận dùng ngưỡng tự phát hiện).
    2. Cắt latent_seq tại [:event_bin+1] cho engine có event -- đúng
       trách nhiệm caller mà data.py Phase II yêu cầu.
    3. Stratified split train/val giữ tỉ lệ event/censored, có phòng hờ
       khi nhóm "event" quá ít (thường chỉ vài engine với FD002).
"""
import numpy as np

from src.phaseII.core.data import EngineRecord


# ============================================================
# BƯỚC 1-2 — Build EngineRecord từ nhãn THẬT (không suy đoán)
# ============================================================

def build_engine_records(
    latent_data: dict[str, np.ndarray],
    true_failure_bin: dict[str, int],
) -> list[EngineRecord]:
    """
    true_failure_bin chỉ chứa engine thuộc nhóm "giữ nguyên" (không cắt).
    Với các engine đó: event_bin = true_failure_bin[engine_id], đúng
    nguyên văn cycle hỏng thật, không qua bất kỳ threshold/heuristic nào.
    Mọi engine KHÔNG có mặt trong true_failure_bin bị coi là censored
    (event_bin=None) -- đúng nhóm "log bình thường, bị cắt trước risk".
    """
    records = []
    for engine_id, z_seq in latent_data.items():
        event_bin = true_failure_bin.get(engine_id)

        if event_bin is not None:
            assert 0 <= event_bin < z_seq.shape[0], (
                f"Engine {engine_id}: event_bin={event_bin} vượt quá "
                f"độ dài latent_seq ({z_seq.shape[0]} bins)"
            )
            latent_seq = z_seq[:event_bin + 1]
            n_bins = event_bin + 1
        else:
            latent_seq = z_seq
            n_bins = z_seq.shape[0]

        records.append(EngineRecord(
            engine_id=str(engine_id),
            latent_seq=latent_seq.astype(np.float32),
            n_bins=n_bins,
            event_bin=event_bin,
        ))
    return records


# ============================================================
# BƯỚC 3 — Stratified split, có phòng hờ nhóm event quá nhỏ
# ============================================================

def _split_group(group: list, train_ratio: float, rng: np.random.Generator):
    """
    Nếu group có ít hơn 2 phần tử, không thể chia có ý nghĩa -- đưa hết
    vào train (cần ít nhất 1 mẫu event trong train để loss hazard có tác
    dụng, xem thảo luận: nếu train_records không có event nào, hazard
    loss không bao giờ được kích hoạt đúng phần quan trọng nhất).
    """
    if len(group) < 2:
        if len(group) == 1:
            warnings.warn(
                "Chỉ có 1 mẫu trong nhóm này -- đưa vào train, "
                "không có mẫu nào cho validation của riêng nhóm này."
            )
        return group, []

    indices = np.arange(len(group))
    rng.shuffle(indices)
    n_train = max(1, int(len(group) * train_ratio))  # đảm bảo train có ít nhất 1
    return (
        [group[i] for i in indices[:n_train]],
        [group[i] for i in indices[n_train:]],
    )


def split_train_val(
    records: list[EngineRecord],
    train_ratio: float = 0.8,
    random_seed: int = 42,
):
    """
    Stratified split giữ tỉ lệ event/censored trong cả train lẫn val.
    Với FD002, nhóm "event" (giữ nguyên) thường RẤT ÍT (vài engine) --
    stratify càng quan trọng để không lỡ dồn hết vào 1 bên; _split_group
    xử lý riêng trường hợp nhóm quá nhỏ để không chia ra 1 bên rỗng.
    """
    rng = np.random.default_rng(random_seed)

    censored = [r for r in records if r.event_bin is None]
    has_event = [r for r in records if r.event_bin is not None]

    if len(has_event) == 0:
        warnings.warn(
            "Không có engine nào có event_bin thật (true_failure_bin rỗng "
            "hoặc không khớp latent_data) -- Phase II sẽ không train được "
            "đúng ý nghĩa (hazard loss cần ít nhất 1 event trong train)."
        )

    train_censored, val_censored = _split_group(censored, train_ratio, rng)
    train_event, val_event = _split_group(has_event, train_ratio, rng)

    train_records = train_censored + train_event
    val_records = val_censored + val_event

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    return train_records, val_records


# ============================================================
# ENTRY POINT
# ============================================================

def prepare_training_data(
    latent_data: dict[str, np.ndarray],
    true_failure_bin: dict[str, int],
    train_ratio: float = 0.8,
    random_seed: int = 42,
):
    records = build_engine_records(latent_data, true_failure_bin)
    return split_train_val(records, train_ratio=train_ratio, random_seed=random_seed)
