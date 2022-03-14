import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Tuple, Optional, Sequence

FileMap = Mapping[int, Tuple[str, int]]

@dataclass
class DatasetIndex:
    data_dir: Path
    file_map: FileMap
    masked_data_dir: Optional[Path]
    num_masked_instances: int
    train_indices: Sequence[int]
    dev_indices: Sequence[int]
    test_indices: Sequence[int]
    interaction_codes: Mapping[str, int]
    tag_codes: Mapping[str, int]
    # num_interactions: int
    # num_tags: int

    @property
    @lru_cache()
    def num_interactions(self) -> int:
        return len(self.interaction_codes)

    @property
    @lru_cache()
    def num_tags(self) -> int:
        return len(self.tag_codes)

    @property
    @lru_cache()
    def codes_tag(self) -> Mapping[int, str]:
        return {v:k for k, v in self.tag_codes}

    @property
    @lru_cache()
    def codes_interaction(self) -> Mapping[int, str]:
        return {v:k for k, v in self.interaction_codes}

    @classmethod
    def from_json_file(cls, path:Path) -> 'DatasetIndex':

        with path.open() as f:
            data = json.load(f)

        return cls(
            data_dir=Path(data["data_dir"]),
            file_map={int(k):v for k, v in data["file_map"].items()},
            masked_data_dir = Path(data["masked_data_dir"]) if "masked_data_dir" in data else None,
            num_masked_instances = data["num_masked_instances"] if "num_masked_instances" in data else 0,
            train_indices= data["train_indices"],
            dev_indices=data["dev_indices"],
            test_indices=data["test_indices"],
            interaction_codes={k:int(v) for k,v in data["interaction_codes"].items()},
            tag_codes = {k: int(v) for k, v in data["tag_codes"].items()}
            # num_interactions=data["num_interactions"],
            # num_tags=data["num_tags"]
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "data_dir": str(self.data_dir),
                "file_map": self.file_map,
                "masked_data_dir": str(self.masked_data_dir),
                "num_masked_instances": self.num_masked_instances,
                "train_indices": list(self.train_indices),
                "dev_indices": list(self.dev_indices),
                "test_indices": list(self.test_indices),
                "interaction_codes": self.interaction_codes,
                "tag_codes": self.tag_codes
                # "num_interactions": self.num_interactions,
                # "num_tags": self.num_tags
            }
        )

    def __hash__(self):
        return 0