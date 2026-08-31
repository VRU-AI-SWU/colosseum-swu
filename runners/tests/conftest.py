from __future__ import annotations


import textwrap
from pathlib import Path

import pytest

import vacuum

REPO = Path(__file__).resolve().parents[2]
CONFIGS = Path(vacuum.__file__).resolve().parent / "configs"


def write_submission(directory: Path, body: str) -> Path:
    """สร้างโฟลเดอร์ submission ที่มี agent.py ตามที่ส่งเข้ามา"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return directory


@pytest.fixture
def make_submission(tmp_path: Path):
    counter = {"n": 0}

    def _make(body: str) -> Path:
        counter["n"] += 1
        return write_submission(tmp_path / f"sub{counter['n']}", body)

    return _make


@pytest.fixture
def baseline_submission(make_submission):
    """submission ที่ห่อ baseline ตัวใดตัวหนึ่ง — ใช้เทียบผลกับการรันแบบ in-process"""

    def _make(level: str) -> Path:
        return make_submission(
            f"""
            from vacuum.baselines import BASELINES

            class Agent:
                def __init__(self, config):
                    self._inner = BASELINES[{level!r}](config)

                def reset(self, episode_info):
                    self._inner.reset(episode_info)

                def act(self, observation):
                    return self._inner.act(observation)
            """
        )

    return _make


# CP462 ไม่มีเมล็ดสำรองให้ตั้งอีกแล้ว — ชุดที่ใช้ตัดสินเป็นไฟล์ในคลัง ไม่ใช่ของที่
# สร้างจากตัวเลข · เทสต์ที่ต้องใช้คลังสร้างคลังชั่วคราวของตัวเอง (ดู
# `test_prediction_cp462.py::tasks`) ซึ่งทำให้มันเดินเส้นทางเดียวกับผู้สอนจริง


# ── โจทย์ CP462 สำหรับเทสต์ — คลังชุดข้อมูลชั่วคราว ─────────────────
#
# ผู้สอนอัปโหลด CSV เข้าคลัง แล้ว config อ้างถึงไฟล์นั้นด้วยลายนิ้วมือ · เทสต์
# ทำแบบเดียวกันเป๊ะ เพื่อไม่ให้มีเส้นทางพิเศษที่ทดสอบบ่อยแต่ไม่มีใครใช้จริง

#: โจทย์ที่ปั๊มไว้ให้เทสต์ใช้ — ค่าตรงกับที่ `tabular.generator` สร้าง
TABULAR_TASKS = {
    "churn": dict(kind="classification", primary="macro_f1", target="churned",
                  labels=[0, 1], drop=["account_id"]),
    "housing": dict(kind="regression", primary="r2", target="monthly_value",
                    labels=[], drop=["account_id"]),
}
TABULAR_ROWS = 4000


@pytest.fixture(scope="session")
def tabular_tasks(tmp_path_factory):
    """`{ชื่อ: (config_path, spec)}` พร้อมคลังที่ `ARENA_DATASETS` ชี้ไปหา

    import ข้างในฟังก์ชันเพราะเครื่องที่ยังไม่ได้ติดตั้ง `envs/cp462-tabular`
    ต้องเก็บเทสต์อื่นได้ตามปกติ — ไฟล์ที่ใช้ fixture นี้มี `importorskip` ของตัวเอง
    """
    import os

    import yaml
    from tabular import store
    from tabular.arena import PLUGIN
    from tabular.config import TaskSpec
    from tabular.generator import sample_csv

    root = tmp_path_factory.mktemp("datasets")
    previous = os.environ.get(store.DATASETS_ENV)
    os.environ[store.DATASETS_ENV] = str(root)

    configs = tmp_path_factory.mktemp("configs")
    built = {}
    for name, fields in TABULAR_TASKS.items():
        digest = PLUGIN.save_dataset(sample_csv(name, seed=20260101, n=TABULAR_ROWS))
        spec = TaskSpec(title=name.title(), dataset=digest,
                        split_seed=7, bootstrap_seed=11, **fields)
        path = configs / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump({
                "title": spec.title, "kind": spec.kind, "primary": spec.primary,
                "dataset": spec.dataset, "target": spec.target, "drop": list(spec.drop),
                "labels": list(spec.labels), "split_seed": spec.split_seed,
                "bootstrap_seed": spec.bootstrap_seed,
            }, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        built[name] = (path, spec)

    yield built

    if previous is None:
        os.environ.pop(store.DATASETS_ENV, None)
    else:
        os.environ[store.DATASETS_ENV] = previous
