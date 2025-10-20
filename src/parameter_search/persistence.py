import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


class HyperparamStore:
    def __init__(
        self,
        dataset_name: str,
        objective: str,
        mode: str,
        base_dir: Path = Path("hyperparams"),
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', received '{mode}'")
        self.dataset_name = dataset_name
        self.objective = objective
        self.mode = mode
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.base_dir / f"{_sanitize_name(dataset_name)}.json"

        self.data: Dict[str, Any] = self._load()
        self.current_run_id: Optional[str] = None
        self.current_run_trials: List[str] = []

    def _load(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            return {
                "dataset": self.dataset_name,
                "objectives": {},
                "last_updated": None,
            }
        with self.file_path.open("r") as f:
            return json.load(f)

    def _objective_section(self) -> Dict[str, Any]:
        objectives = self.data.setdefault("objectives", {})
        section = objectives.setdefault(
            self.objective,
            {
                "trials": {},
                "best_history": None,
                "best_last_run": None,
                "runs": [],
            },
        )
        section.setdefault("trials", {})
        section.setdefault("best_history", None)
        section.setdefault("best_last_run", None)
        section.setdefault("runs", [])
        return section

    def start_run(self) -> str:
        if self.current_run_id is not None:
            raise RuntimeError("A run is already in progress.")
        run_id = f"{_iso_now()}-{self.objective}"
        self.current_run_id = run_id
        self.current_run_trials = []
        return run_id

    def _ensure_run(self) -> None:
        if self.current_run_id is None:
            raise RuntimeError("No active run. Call start_run() first.")

    @staticmethod
    def canonical_key(params: Dict[str, Any]) -> str:
        return json.dumps(params, sort_keys=True)

    def get_trial(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self.canonical_key(params)
        return self._objective_section()["trials"].get(key)

    def has_successful_trial(self, params: Dict[str, Any]) -> bool:
        record = self.get_trial(params)
        return bool(record and record.get("status") == "ok")

    def best_history(self) -> Optional[Dict[str, Any]]:
        record = self._objective_section().get("best_history")
        return copy.deepcopy(record) if record is not None else None

    def best_last_run(self) -> Optional[Dict[str, Any]]:
        record = self._objective_section().get("best_last_run")
        return copy.deepcopy(record) if record is not None else None

    def _is_better(self, new_value: float, current_value: Optional[float]) -> bool:
        if current_value is None:
            return True
        if self.mode == "min":
            return new_value < current_value
        return new_value > current_value

    def record_success(
        self,
        params: Dict[str, Any],
        objective_value: float,
        metrics: Dict[str, Any],
        objective_key: str,
        seed: int,
        trial_index: int,
    ) -> Dict[str, Any]:
        self._ensure_run()
        section = self._objective_section()
        key = self.canonical_key(params)
        timestamp = _iso_now()
        record = {
            "key": key,
            "params": params,
            "status": "ok",
            "objective": objective_value,
            "objective_key": objective_key,
            "metrics": metrics,
            "seed": seed,
            "run_id": self.current_run_id,
            "trial_index": trial_index,
            "timestamp": timestamp,
        }
        section["trials"][key] = record
        self.current_run_trials.append(key)

        best_history = section.get("best_history")
        best_value = best_history.get("objective") if best_history else None
        if self._is_better(objective_value, best_value):
            section["best_history"] = copy.deepcopy(record)

        self._touch()
        self._save()
        return copy.deepcopy(record)

    def record_failure(
        self,
        params: Dict[str, Any],
        error: str,
        seed: int,
        trial_index: int,
    ) -> Dict[str, Any]:
        self._ensure_run()
        section = self._objective_section()
        key = self.canonical_key(params)
        record = {
            "key": key,
            "params": params,
            "status": "failed",
            "error": error,
            "seed": seed,
            "run_id": self.current_run_id,
            "trial_index": trial_index,
            "timestamp": _iso_now(),
        }
        section["trials"][key] = record
        self.current_run_trials.append(key)
        self._touch()
        self._save()
        return copy.deepcopy(record)

    def finalize_run(self, best_run_record: Optional[Dict[str, Any]]) -> None:
        self._ensure_run()
        section = self._objective_section()
        timestamp = _iso_now()

        run_summary = {
            "run_id": self.current_run_id,
            "timestamp": timestamp,
            "executed_trials": list(self.current_run_trials),
            "best_key": best_run_record.get("key") if best_run_record else None,
        }
        section.setdefault("runs", []).append(run_summary)

        if best_run_record is None:
            best_last_run = {
                "status": "not_run",
                "run_id": self.current_run_id,
                "timestamp": timestamp,
            }
        else:
            best_last_run = copy.deepcopy(best_run_record)

        best_last_run["run_id"] = self.current_run_id
        best_last_run["timestamp"] = timestamp
        section["best_last_run"] = best_last_run

        self.current_run_id = None
        self.current_run_trials = []
        self._touch()
        self._save()

    def _touch(self) -> None:
        self.data["last_updated"] = _iso_now()

    def _save(self) -> None:
        tmp_path = self.file_path.with_suffix(".json.tmp")
        with tmp_path.open("w") as f:
            json.dump(self.data, f, indent=2)
        tmp_path.replace(self.file_path)
