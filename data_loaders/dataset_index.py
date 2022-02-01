import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple, Optional

FileMap = Mapping[int, Tuple[str, int]]

@dataclass
class DatasetIndex:
    data_dir: Path
    file_map: FileMap
    masked_data_dir: Optional[Path]
    num_masked_instances: int

    @classmethod
    def from_json_file(cls, path:Path) -> 'DatasetIndex':

        with path.open() as f:
            data = json.load(f)

        return cls(
            data_dir=Path(data["data_dir"]),
            file_map={int(k):v for k, v in data["file_map"].items()},
            masked_data_dir = Path(data["masked_data_dir"]) if "masked_data_dir" in data else None,
            num_masked_instances = data["num_masked_instances"] if "num_masked_instances" in data else 0
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "data_dir": str(self.data_dir),
                "file_map": self.file_map,
                "masked_data_dir": str(self.masked_data_dir),
                "num_masked_instances": self.num_masked_instances
            }
        )
