"""Re-execute the Rev-2 admission dry-run against the REAL vendored Gate.

CPU fake workers only; no GPU request is made by this script. Exercises the
WINDOWS_REAL_DOCUMENT_POLICY.md gate matrix on the actual PR #28 contract
(src/unlimited_ocr_2_rocm/benchmark/admission.py) supervised by the delivered
production NativeJob, including the re-review additions: TOCTOU (policy and
receipt swap), concurrent ledger instances, and ordered-pages identity.

Windows-only (native Job supervision); exits non-zero on any unexpected case.
Case order matters: refusals run while the device ledger is idle, because a
RUNNING/EVALUATING stage refuses everything else (that refusal is its own case,
deliberately last).
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

if os.name != "nt":
    print(json.dumps({"status": "SKIPPED_NON_WINDOWS"}))
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from unlimited_ocr_2_rocm.benchmark.admission import Gate, Rejected  # noqa: E402
from unlimited_ocr_2_rocm.benchmark.manifest import digest, file_hash  # noqa: E402
from unlimited_ocr_2_rocm.benchmark.windows_job import NativeJob  # noqa: E402

GOLDEN = "correctness_golden"
ARM = "real_document_arm1"
PROBE = "refusal_probe"
GHOST = "ghost_stage_missing_from_policy"
CRASH = "crash_arm"
ARM3 = "real_document_arm3"
FAKE_WORKER = """import sys, time
if '--crash' in sys.argv:
    raise SystemExit(1)
"""
# The frozen worker environment is the controller's full environment plus the
# CLI's two normalized flags — exactly what experiments/admission/run.py pins.
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")


def plan(root: Path, extra: str = "") -> dict:
    argv = [sys.executable, str(root / "entry.py"), str(root / "fake_worker.py")]
    if extra:
        argv.append(extra)
    return {
        "argv": argv,
        "cwd": str(root),
        "env": dict(ENV),
        "dependency": GOLDEN,
    }


def build(root: Path) -> dict:
    (root / "fake_worker.py").write_text(FAKE_WORKER, encoding="utf-8")
    entry_body = "import runpy, sys\nrunpy.run_path(sys.argv[1])\n"
    (root / "entry.py").write_text(entry_body, encoding="utf-8")
    (root / "torch-wheel.txt").write_text(
        "sha256:415d2dbd4b6ab002cff1c54795fdd66322abd9a46bf982cb64eebb4493ac2e7c\n",
        encoding="utf-8",
    )
    (root / "page-000.png").write_bytes(b"fake-page-1")
    (root / "page-001.png").write_bytes(b"fake-page-2")
    ordered = root / "ordered-paths.txt"
    ordered.write_text(f"{root / 'page-000.png'}\n{root / 'page-001.png'}\n", encoding="utf-8")
    (root / "code-note.txt").write_text("frozen source digest placeholder\n", encoding="utf-8")
    (root / "admission-receipt.json").write_text("frozen pre-admission receipt\n", encoding="utf-8")
    runner = ROOT / "experiments/admission/run.py"
    module = ROOT / "src/unlimited_ocr_2_rocm/benchmark/admission.py"
    policy = root / "policy.json"
    golden_plan = {
        "argv": [sys.executable, str(root / "entry.py"), str(root / "fake_worker.py")],
        "cwd": str(root),
        "env": dict(ENV),
        "dependency": None,
    }
    policy.write_text(
        json.dumps(
            {
                GOLDEN: golden_plan,
                ARM: plan(root),
                PROBE: plan(root),
                CRASH: plan(root, "--crash"),
                ARM3: plan(root),
            }
        ),
        encoding="utf-8",
    )
    return {
        "sources": {
            str(runner): file_hash(runner),
            str(module): file_hash(module),
            str(root / "entry.py"): file_hash(root / "entry.py"),
            str(root / "fake_worker.py"): file_hash(root / "fake_worker.py"),
            str(root / "code-note.txt"): file_hash(root / "code-note.txt"),
            sys.executable: file_hash(Path(sys.executable)),
        },
        "wheels": {str(root / "torch-wheel.txt"): file_hash(root / "torch-wheel.txt")},
        "fixtures": {
            str(root / "page-000.png"): file_hash(root / "page-000.png"),
            str(root / "page-001.png"): file_hash(root / "page-001.png"),
            str(ordered): file_hash(ordered),
        },
        "policy": {str(policy): file_hash(policy)},
        "artifacts": {
            str(root / "admission-receipt.json"): file_hash(root / "admission-receipt.json")
        },
        "target": "windows/gfx1151",
    }


def plan_for(policy_path: Path, stage: str) -> list[str]:
    return json.loads(policy_path.read_text(encoding="utf-8"))[stage]["argv"]


def dispatch(gate: Gate, stage: str, identity: dict) -> str:
    """Dispatch through a fresh owned NativeJob; returns terminal ledger state."""
    policy = json.loads(Path(next(iter(identity["policy"]))).read_text(encoding="utf-8"))
    job = NativeJob()
    try:
        gate.dispatch(
            stage,
            policy[stage]["dependency"],
            identity,
            job,
            timeout=8,
            cwd=policy[stage]["cwd"],
            env=policy[stage]["env"],
        )
    finally:
        job.close()
    return gate.state(stage)


def evaluate_pass(gate: Gate, stage: str, identity: dict, root: Path) -> Path:
    execution = gate.execution(stage)
    output = root / f"{stage}-output.md"
    output.write_text("sealed worker output\n", encoding="utf-8")
    receipt = root / f"{stage}-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "identity": digest(identity),
                "execution": execution,
                "status": "PASS",
                "complete": True,
                "strict_correctness": True,
                "artifacts": {str(output): file_hash(output)},
            }
        ),
        encoding="utf-8",
    )
    gate.evaluate(stage, receipt, file_hash(receipt))
    return receipt


def main() -> int:
    results: list[dict] = []

    def record(case: str, result: str, **extra) -> None:
        row = {"case": case, "result": result, **extra}
        results.append(row)
        print(f"{case}: {result}", file=sys.stderr)

    def expect_reject(case: str, needle: str, action) -> None:
        job = NativeJob()
        try:
            action(job)
            record(case, "NOT_REFUSED", expected=needle)
        except Rejected as error:
            message = str(error)
            record(
                case,
                "REFUSED",
                reason=message,
                matches=needle in message,
                workers_spawned=job.accounting()["total_processes"],
            )
        finally:
            job.close()

    with tempfile.TemporaryDirectory(prefix="uocr2-real-gate-") as tmp:
        root = Path(tmp)
        identity = build(root)
        gate = Gate(root / "ledger.sqlite3")
        policy_path = Path(next(iter(identity["policy"])))

        # -- 1. fresh golden qualification admits and passes (policy gate 8,
        #       positive side: this is the P6 slot on the real device ledger)
        golden_argv = json.loads(policy_path.read_text(encoding="utf-8"))[GOLDEN]["argv"]
        gate.prepare(GOLDEN, identity, golden_argv, initial_qualification=True)
        try:
            state = dispatch(gate, GOLDEN, identity)
            receipt = evaluate_pass(gate, GOLDEN, identity, root)
            record(
                "initial_qualification_admit_pass",
                gate.state(GOLDEN),
                worker_state_after_dispatch=state,
                receipt=file_hash(receipt),
            )
        except Rejected as error:
            record("initial_qualification_admit_pass", "REJECTED", reason=str(error))

        # -- 2. refusals on a reusable PREPARED probe stage (failures roll back
        #       before RUNNING, so the probe stays PREPARED for the next case)
        gate.prepare(PROBE, identity, plan_for(policy_path, "PROBE"))

        def tampered(group: str, path: Path, action, restore) -> None:
            original = path.read_bytes()
            action(path)
            try:
                expect_reject(
                    f"refuse_{group}_identity",
                    f"Unverified {group} artifact",
                    lambda job: gate.dispatch(
                        PROBE,
                        GOLDEN,
                        identity,
                        job,
                        timeout=8,
                        cwd=plan(root)["cwd"],
                        env=plan(root)["env"],
                    ),
                )
            finally:
                restore(path, original)

        source = root / "code-note.txt"
        tampered(
            "sources",
            source,
            lambda p: p.write_text("mutated\n", encoding="utf-8"),
            lambda p, o: p.write_bytes(o),
        )
        wheel = root / "torch-wheel.txt"
        tampered(
            "wheels",
            wheel,
            lambda p: p.write_text("candidate-6a67d158\n", encoding="utf-8"),
            lambda p, o: p.write_bytes(o),
        )
        page = root / "page-000.png"
        tampered(
            "fixtures",
            page,
            lambda p: p.write_bytes(b"tampered-page"),
            lambda p, o: p.write_bytes(o),
        )
        ordered = root / "ordered-paths.txt"
        original_ordered = ordered.read_bytes()

        def swap_order(p: Path) -> None:
            lines = p.read_text(encoding="utf-8").splitlines()
            p.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

        swap_order(ordered)
        expect_reject(
            "refuse_ordered_pages_swapped",
            "Unverified fixtures artifact",
            lambda job: gate.dispatch(
                PROBE,
                GOLDEN,
                identity,
                job,
                timeout=8,
                cwd=plan(root)["cwd"],
                env=plan(root)["env"],
            ),
        )
        ordered.write_bytes(original_ordered)

        # policy file swapped between prepare and dispatch (TOCTOU)
        original_policy = policy_path.read_bytes()
        evil = json.loads(original_policy.decode("utf-8"))
        evil[PROBE]["argv"] = [sys.executable, "imposter.py"]
        policy_path.write_text(json.dumps(evil), encoding="utf-8")
        expect_reject(
            "refuse_policy_swapped_toctou",
            "Changed experiment policy",
            lambda job: gate.dispatch(
                PROBE,
                GOLDEN,
                identity,
                job,
                timeout=8,
                cwd=plan(root)["cwd"],
                env=plan(root)["env"],
            ),
        )
        policy_path.write_bytes(original_policy)

        # golden receipt swapped after evaluation (TOCTOU; gate 7)
        golden_receipt = root / f"{GOLDEN}-receipt.json"
        original_receipt = golden_receipt.read_bytes()
        golden_receipt.write_text("swapped\n", encoding="utf-8")
        expect_reject(
            "refuse_receipt_swapped_toctou",
            "Qualification changed after evaluation",
            lambda job: gate.dispatch(
                PROBE,
                GOLDEN,
                identity,
                job,
                timeout=8,
                cwd=plan(root)["cwd"],
                env=plan(root)["env"],
            ),
        )
        golden_receipt.write_bytes(original_receipt)

        # sealed qualification artifact mutated (gate 7 artifact half)
        artifact = root / f"{GOLDEN}-output.md"
        original_artifact = artifact.read_bytes()
        artifact.write_text("mutated output\n", encoding="utf-8")
        expect_reject(
            "refuse_qualification_artifact_mutated",
            "Qualification artifact changed",
            lambda job: gate.dispatch(
                PROBE,
                GOLDEN,
                identity,
                job,
                timeout=8,
                cwd=plan(root)["cwd"],
                env=plan(root)["env"],
            ),
        )
        artifact.write_bytes(original_artifact)

        # tampered identity manifest (wrong hash passed, files intact)
        stale_identity = json.loads(json.dumps(identity))
        stale_identity["sources"][str(root / "code-note.txt")] = "0" * 64
        expect_reject(
            "refuse_stale_identity_manifest",
            "Stale or already used stage",
            lambda job: gate.dispatch(
                PROBE,
                GOLDEN,
                stale_identity,
                job,
                timeout=8,
                cwd=plan(root)["cwd"],
                env=plan(root)["env"],
            ),
        )

        # stage absent from the frozen policy cannot even be prepared (gate 6)
        expect_reject(
            "refuse_stage_missing_from_policy",
            "Missing or invalid stage policy",
            lambda job: gate.prepare(GHOST, identity, [sys.executable, "ghost.py"]),
        )

        # dependent work cannot omit its qualification (gate 8 negative side)
        expect_reject(
            "refuse_dependency_omitted",
            "differs from frozen policy",
            lambda job: gate.prepare(
                ARM,
                identity,
                json.loads(policy_path.read_text(encoding="utf-8"))[ARM]["argv"],
                initial_qualification=True,
            ),
        )

        # -- 3. worker crash is terminal FAIL and blocks its stage (semantics)
        gate.prepare(CRASH, identity, plan_for(policy_path, "CRASH"))
        expect_reject(
            "worker_crash_is_fail",
            "Worker failed; no correctness qualification",
            lambda job: gate.dispatch(
                CRASH,
                GOLDEN,
                identity,
                job,
                timeout=8,
                cwd=plan(root, "--crash")["cwd"],
                env=plan(root)["env"],
            ),
        )
        record("worker_crash_state", gate.state(CRASH))
        expect_reject(
            "resume_after_fail_refused",
            "Stale or already used stage",
            lambda job: gate.dispatch(
                CRASH,
                GOLDEN,
                identity,
                job,
                timeout=8,
                cwd=plan(root)["cwd"],
                env=plan(root)["env"],
            ),
        )

        # -- 4. concurrent ledger instance is refused (re-review addition)
        lock = sqlite3.connect(gate.path, isolation_level=None)
        lock.execute("BEGIN IMMEDIATE")
        try:
            gate.state(GOLDEN)
            record("concurrent_ledger_busy", "NOT_REFUSED")
        except Rejected as error:
            record(
                "concurrent_ledger_busy", "REFUSED", reason=str(error), matches="Busy" in str(error)
            )
        finally:
            lock.commit()
            lock.close()

        # -- 5. the real-document arm admits after golden PASS (gate 8 positive)
        gate.prepare(ARM, identity, plan_for(policy_path, "ARM"))
        try:
            state = dispatch(gate, ARM, identity)
            record(
                "dependent_arm_admit_after_golden",
                state,
                note="EVALUATING = admitted; evaluator seals after this dry-run",
            )
        except Rejected as error:
            record("dependent_arm_admit_after_golden", "REJECTED", reason=str(error))

        # -- 6. with the device busy, a third experiment is refused (gate 9)
        gate.prepare(ARM3, identity, plan_for(policy_path, "ARM3"))
        expect_reject(
            "refuse_unfinished_device_work",
            "Unfinished device work",
            lambda job: gate.dispatch(
                ARM3, GOLDEN, identity, job, timeout=8, cwd=plan(root)["cwd"], env=plan(root)["env"]
            ),
        )
        record(
            "final_ledger_states",
            "OK",
            **{name: gate.state(name) for name in (GOLDEN, ARM, CRASH, PROBE)},
        )

    refusals = [r for r in results if r["result"] == "REFUSED"]
    mismatches = [r["case"] for r in refusals if not r.get("matches", True)]
    not_refused = [r["case"] for r in results if r["result"] == "NOT_REFUSED"]
    status = "DRYRUN_PASS" if not mismatches and not not_refused else "DRYRUN_FAIL"
    receipt = {
        "schema": 2,
        "mode": "CPU_FAKE_WORKERS_ONLY",
        "gate": "REAL vendored PR#28 Gate (src/unlimited_ocr_2_rocm/benchmark/admission.py)",
        "supervisor": "production NativeJob (src/unlimited_ocr_2_rocm/benchmark/windows_job.py)",
        "gpu_executed": False,
        "cases": results,
        "summary": {
            "cases_total": len(results),
            "refusals_verified": len(refusals),
            "reason_needle_mismatches": mismatches,
            "not_refused": not_refused,
        },
        "status": status,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("admission-real-gate-dryrun.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(receipt["summary"]))
    return 0 if status == "DRYRUN_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
