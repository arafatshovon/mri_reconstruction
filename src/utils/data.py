from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def load_file_list_from_csv(
    csv_path: str,
    acquisition: Optional[Sequence[str]] = None,
    acquisition_index: Optional[int] = None,
    data_root: Optional[str] = None,
) -> List[str]:
    df = pd.read_csv(csv_path)
    if acquisition is not None:
        if acquisition_index is None:
            acquisition_index = 0
        target_acq = acquisition[acquisition_index]
        df = df[df["acquisition"] == target_acq]

    files = df["file_name"].values.tolist()
    if data_root:
        root = Path(data_root)
        files = [str(p if Path(p).is_absolute() else root / p) for p in files]
    return files
