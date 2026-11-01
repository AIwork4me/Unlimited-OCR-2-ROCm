"""Opt-in experiment admission. No inference imports or production activation.

One SQLite ledger per exclusively owned physical device. A crashed RUNNING or
EVALUATING stage blocks future work; recovery never invents a PASS. The trusted
controller/evaluator owns the ledger and immutable files, not hostile writers.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from .manifest import digest, file_hash, safe_id


class Rejected(RuntimeError):
    """No dependent worker may start."""


class Supervisor(Protocol):
    def accounting(self) -> dict[str, int]: ...
    def spawn(self, command: list[str], **kwargs: Any) -> subprocess.Popen: ...
    def terminate(self) -> None: ...


def _verify(identity: dict[str, Any]) -> None:
    if set(identity) != {"sources", "wheels", "fixtures", "policy", "artifacts", "target"}:
        raise Rejected("Incomplete experiment identity")
    if identity["target"] != "windows/gfx1151":
        raise Rejected("Unsupported physical target")
    for group in ("sources", "wheels", "fixtures", "policy", "artifacts"):
        entries = identity[group]
        if not isinstance(entries, dict) or not entries:
            raise Rejected(f"Missing {group} identity")
        for name, expected in entries.items():
            if not Path(name).is_absolute() or file_hash(Path(name)) != expected:
                raise Rejected(f"Unverified {group} artifact")


def _plan(identity: dict[str, Any], stage: str) -> dict[str, Any]:
    try:
        policies = identity["policy"]
        if len(policies) != 1:
            raise Rejected("One frozen stage policy required")
        path, expected = next(iter(policies.items()))
        if file_hash(Path(path)) != expected:
            raise Rejected("Changed experiment policy")
        result: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))[stage]
        if (
            not isinstance(result, dict)
            or set(result) != {"argv", "cwd", "env", "dependency"}
            or not isinstance(result["env"], dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in result["env"].items())
            or not isinstance(result["cwd"], str)
            or not Path(result["cwd"]).is_absolute()
            or not isinstance(result["argv"], list)
            or not (result["dependency"] is None or isinstance(result["dependency"], str))
        ):
            raise Rejected("Incomplete execution policy")
        return result
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise Rejected("Missing or invalid stage policy") from error


def _execution(row: sqlite3.Row) -> str:
    return digest(
        {k: row[k] for k in ("name", "identity", "command", "attempt", "admission", "engine")}
    )


class Gate:
    def __init__(self, path: Path) -> None:
        self.path = path
        with self._transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS stages (
                name TEXT PRIMARY KEY, identity TEXT NOT NULL, command TEXT NOT NULL,
                state TEXT NOT NULL, receipt TEXT, receipt_hash TEXT, execution_ok INTEGER,
                admission TEXT, initial INTEGER NOT NULL, engine TEXT NOT NULL, attempt TEXT)""")

    @contextmanager
    def _transaction(self, *, lock_timeout: float = 0) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=lock_timeout, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except sqlite3.Error as error:
            raise Rejected("Busy or invalid admission ledger") from error
        finally:
            db.close()  # Uncommitted work rolls back; durable RUNNING survives.

    def _row(self, db: sqlite3.Connection, stage: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM stages WHERE name = ?", (stage,)).fetchone()
        if row is None:
            raise Rejected("Missing qualification/stage")
        if row["engine"] != file_hash(Path(__file__)):
            raise Rejected("Admission implementation changed")
        return row

    def state(self, stage: str) -> str:
        with self._transaction() as db:
            return str(self._row(db, stage)["state"])

    def execution(self, stage: str) -> str:
        """Identity the evaluator must bind, specific to this execution attempt."""
        with self._transaction() as db:
            return _execution(self._row(db, stage))

    def prepare(
        self,
        stage: str,
        identity: dict[str, Any],
        command: list[str],
        *,
        initial_qualification: bool = False,
    ) -> None:
        safe_id(stage)
        if not command or not all(isinstance(arg, str) and arg for arg in command):
            raise Rejected("Frozen argument list required")
        plan = _plan(identity, stage)
        if plan.get("argv") != command or (plan.get("dependency") is None) != initial_qualification:
            raise Rejected("Command or qualification purpose differs from frozen policy")
        with self._transaction() as db:
            db.execute(
                "INSERT INTO stages(name,identity,command,state,initial,engine) "
                "VALUES(?,?,?,?,?,?)",
                (
                    stage,
                    json.dumps(identity, sort_keys=True),
                    json.dumps(command),
                    "PREPARED",
                    int(initial_qualification),
                    file_hash(Path(__file__)),
                ),
            )

    def evaluate(self, stage: str, receipt: Path, expected_hash: str) -> None:
        """Commit the trusted evaluator's sealed result; PARTIAL is terminal FAIL."""
        with self._transaction() as db:
            row = self._row(db, stage)
            if row["state"] != "EVALUATING":
                raise Rejected("Only completed execution can be evaluated")
            identity = json.loads(row["identity"])
            try:
                _verify(identity)
                data = json.loads(receipt.read_text(encoding="utf-8"))
                if file_hash(receipt) != expected_hash:
                    raise Rejected("Unverified qualification receipt")
                artifacts = data.get("artifacts")
                passed = (
                    row["execution_ok"] == 1
                    and data.get("identity") == digest(identity)
                    and data.get("execution") == _execution(row)
                    and data.get("status") == "PASS"
                    and data.get("complete") is True
                    and data.get("strict_correctness") is True
                    and isinstance(artifacts, dict)
                    and bool(artifacts)
                )
                if passed:
                    for name, expected in artifacts.items():
                        if not Path(name).is_absolute() or file_hash(Path(name)) != expected:
                            passed = False
            except (OSError, ValueError, TypeError) as error:
                raise Rejected("Missing or invalid evaluation") from error
            db.execute(
                "UPDATE stages SET state=?,receipt=?,receipt_hash=? WHERE name=?",
                ("PASS" if passed else "FAIL", str(receipt.resolve()), expected_hash, stage),
            )

    def _admit(
        self, db: sqlite3.Connection, dependency: str | None, identity: dict[str, Any]
    ) -> str:
        try:
            _verify(identity)
            if dependency is None:
                return "ROOT_QUALIFICATION_ONLY"
            row = self._row(db, dependency)
            if row["state"] != "PASS" or json.loads(row["identity"]) != identity:
                raise Rejected("Current complete strict PASS required")
            receipt = Path(row["receipt"])
            if file_hash(receipt) != row["receipt_hash"]:
                raise Rejected("Qualification changed after evaluation")
            for name, expected in json.loads(receipt.read_text(encoding="utf-8"))[
                "artifacts"
            ].items():
                if file_hash(Path(name)) != expected:
                    raise Rejected("Qualification artifact changed")
            return str(row["receipt_hash"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise Rejected("Missing or invalid admission artifacts") from error

    def dispatch(
        self,
        stage: str,
        dependency: str | None,
        identity: dict[str, Any],
        supervisor: Supervisor,
        *,
        timeout: float,
        **kwargs: Any,
    ) -> None:
        """Reserve durably, recheck under lock, spawn the frozen command and drain.

        None is exclusively for an initial correctness qualification, never for
        performance/population expansion. All callers share this device ledger.
        Only an idle, owned supervisor may be passed. Close its handle afterward.
        """
        if not math.isfinite(timeout) or timeout <= 0:
            raise Rejected("A finite positive worker timeout is required")
        with self._transaction() as db:
            row = self._row(db, stage)
            if row["state"] != "PREPARED" or json.loads(row["identity"]) != identity:
                raise Rejected("Stale or already used stage")
            if dependency is None and not row["initial"]:
                raise Rejected("Dependent work cannot omit qualification")
            plan = _plan(identity, stage)
            command = json.loads(row["command"])
            if (
                plan.get("argv") != command
                or plan.get("dependency") != dependency
                or set(kwargs) - {"cwd", "env", "stdout", "stderr"}
                or str(Path(kwargs.get("cwd", Path.cwd())).resolve()) != plan.get("cwd")
                or kwargs.get("env", dict(os.environ)) != plan.get("env")
            ):
                raise Rejected("Execution arguments/cwd/environment differ from policy")
            if (
                len(command) < 2
                or command[0] not in identity["sources"]
                or command[1] not in identity["sources"]
            ):
                raise Rejected("Executable and entry script require source hashes")
            kwargs.update(cwd=plan["cwd"], env=dict(plan["env"]))
            if db.execute(
                "SELECT 1 FROM stages WHERE state IN ('RUNNING','EVALUATING')"
            ).fetchone():
                raise Rejected("Unfinished device work requires explicit investigation")
            admission = self._admit(db, dependency, identity)
            if supervisor.accounting()["active_processes"] != 0:
                raise Rejected("Unsafe active children")
            db.execute(
                "UPDATE stages SET state='RUNNING',admission=?,attempt=? WHERE name=?",
                (admission, uuid.uuid4().hex, stage),
            )
        # No reusable authorization token escapes this function. A restart sees
        # RUNNING even if interrupted here, before a process exists.
        failure: BaseException | None = None
        # The reserved owner tolerates a short metadata transaction in the gap;
        # ordinary contenders still fail immediately. Timeout remains fail-closed.
        with self._transaction(lock_timeout=5) as db:
            if self._admit(db, dependency, identity) != admission:
                raise Rejected("Admission changed before spawn")
            if supervisor.accounting()["active_processes"] != 0:
                raise Rejected("Unsafe active children")
            # Re-read after accounting, immediately before spawn. Hold the DB
            # lock through worker exit/cleanup, preventing evaluator/launch races.
            self._admit(db, dependency, identity)
            ok = False
            try:
                process = supervisor.spawn(json.loads(row["command"]), **kwargs)
                ok = process.wait(timeout=timeout) == 0
            except BaseException as error:
                failure = error
            finally:
                supervisor.terminate()
                ok = ok and supervisor.accounting()["active_processes"] == 0
                db.execute(
                    "UPDATE stages SET state=?,execution_ok=? WHERE name=?",
                    ("EVALUATING" if ok else "FAIL", int(ok), stage),
                )
        if failure is not None:
            raise failure
        if not ok:
            raise Rejected("Worker failed; no correctness qualification")
