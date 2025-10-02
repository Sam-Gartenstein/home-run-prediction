from __future__ import annotations
import json, logging, hashlib, re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class RunInfo:
    run_id: str
    started_at: str
    ended_at: str | None = None

class RunLogger:
    """
    Creates runs like:
      artifacts/logs/2025-10-01_01/{config.yaml, metrics.jsonl, events.log, [notes.md]}
    """

    def __init__(self, base_dir: Path, run_name: Optional[str] = None, width: int = 2):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = self.next_run_name(self.base_dir, width=width)
        else:
            # ensure uniqueness if user-supplied name exists
            candidate = self.base_dir / run_name
            suffix = 2
            while candidate.exists():
                candidate = self.base_dir / f"{run_name}-{suffix}"
                suffix += 1
            run_name = candidate.name

        self.run_dir = self.base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)

        # set up logging
        self._logger = logging.getLogger(f"runlog_{id(self)}")
        self._logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.run_dir / "events.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
        self._logger.addHandler(fh)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.info = RunInfo(run_id=self.run_dir.name, started_at=datetime.utcnow().isoformat() + "Z")

    # -------- ID helpers --------
    @classmethod
    def next_run_name(cls, base_dir: Path, date_str: Optional[str] = None, width: int = 2) -> str:
        """
        Return next available name 'YYYY-MM-DD_XX' directly under base_dir.
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        prefix = f"{date_str}_"
        pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")

        max_n = 0
        for p in base_dir.iterdir():
            if p.is_dir():
                m = pat.match(p.name)
                if m:
                    try:
                        n = int(m.group(1))
                        if n > max_n:
                            max_n = n
                    except ValueError:
                        pass
        return f"{date_str}_{max_n + 1:0{width}d}"

    # ------------ context manager ------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()

    # ------------ basics ------------
    def start(self):
        self._write_record({"event": "start", **asdict(self.info)})

    def finish(self):
        self.info.ended_at = datetime.utcnow().isoformat() + "Z"
        self._write_record({"event": "end", **asdict(self.info)})
        self._logger.info("run finished")

    def log_event(self, msg: str, level: str = "info"):
        getattr(self._logger, level, self._logger.info)(msg)
        self._write_record({"event": "log", "message": msg})

    def log_config(self, cfg: dict):
        try:
            import yaml
            (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        except Exception:
            (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        self._write_record({"event": "config", "config": cfg})

    def log_metrics(self, metrics: dict):
        self._write_record({"event": "metrics", "metrics": metrics})

    def attach_idata(self, path: Path):
        sha256 = None
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            sha256 = h.hexdigest()
        except Exception:
            pass
        self._write_record({"event": "idata", "path": str(path), "sha256": sha256})

    # ------------ notes (optional) ------------
    def log_note(self, note: str):
        self.log_event(note)
        self._write_record({"event": "note", "note": note})

    def write_notes_md(self, text: str, append: bool = True):
        p = self.run_dir / "notes.md"
        mode = "a" if append and p.exists() else "w"
        with open(p, mode) as f:
            if mode == "a":
                f.write("\n")
            f.write(text.strip() + "\n")
        self._write_record({"event": "artifact", "path": str(p), "kind": "notes.md"})

    # ------------ convenience ------------
    @property
    def dir(self) -> Path:
        return self.run_dir

    def _write_record(self, obj: dict):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    @classmethod
    def open(cls, run_dir: Path):
        """Attach to an existing run folder."""
        self = cls.__new__(cls)  # bypass __init__
        self.base_dir = Path(run_dir).parent
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(f"runlog_{id(self)}")
        self._logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.run_dir / "events.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
        self._logger.addHandler(fh)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.info = RunInfo(run_id=self.run_dir.name, started_at=datetime.utcnow().isoformat() + "Z")
        return self

