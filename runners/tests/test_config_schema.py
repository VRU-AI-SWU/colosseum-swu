"""หน้าตาของ config ที่หน้าเว็บใช้สร้างฟอร์ม — ต้องตรงกับ config จริงเสมอ

ฟอร์มที่สร้างจาก schema ผิด **ไม่ได้พังทันที** — มันรับค่าที่ loader ปฏิเสธ
แล้วผู้สอนที่กรอกครบทุกช่องกดบันทึกไม่ได้ โดยข้อความที่เห็นเป็นข้อความของ loader
ซึ่งพูดถึงฟิลด์ที่ฟอร์มไม่ได้แสดงด้วยซ้ำ · หรือแย่กว่านั้นคือตกฟิลด์ไปเงียบๆ
แล้วได้ config ที่ไม่ครบ

เทสต์ในไฟล์นี้จึงผูกสามอย่างเข้าด้วยกัน

  1. schema ต้องครอบ **ทุกฟิลด์** ของ config จริง (อนุมานจาก dataclass จึงได้ฟรี
     แต่ยังต้องยืนยันว่าการอนุมานถูก)
  2. schema ต้องครอบ **ทุกคีย์ที่ปรากฏใน YAML ที่แจกจริง** — ไฟล์ที่ใช้อยู่คือ
     หลักฐานว่าคีย์ไหนมีจริง
  3. **ขอบเขตที่ประกาศต้องตรงกับที่ loader บังคับจริง** — ยิงค่านอกขอบเขตเข้าไป
     แล้วต้องถูกปฏิเสธ ไม่ใช่เชื่อว่าคนเขียนจำถูก
"""

from __future__ import annotations

import pytest
import yaml

from runners.sandbox.schema import Limit, derive

# ── ตัวช่วยที่ใช้ร่วมกันทุก env ──────────────────────────────────────


def keys_of(schema) -> set[str]:
    return {f["key"] for f in schema}


def yaml_keys(path) -> set[str]:
    """คีย์แบบ dotted ที่ปรากฏจริงในไฟล์ YAML"""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = set()
    for key, value in data.items():
        if isinstance(value, dict):
            out |= {f"{key}.{name}" for name in value}
        else:
            out.add(key)
    return out


# ── ตัวอนุมานเอง ────────────────────────────────────────────────────


def test_a_field_without_a_default_is_required_not_an_error():
    """config บางตัวบังคับให้ประกาศเอง (CP462) บางตัวมีค่าเริ่มต้นครบ (CP463)
    ทั้งสองแบบต้องอธิบายเป็นฟอร์มได้"""
    from dataclasses import dataclass, field

    @dataclass
    class Sample:
        must_fill: str
        has_default: int = 3
        listy: list = field(default_factory=list)

    got = {f.key: f for f in derive(Sample)}
    assert got["must_fill"].required and got["must_fill"].default is None
    assert not got["has_default"].required and got["has_default"].default == 3
    assert got["listy"].default == []


def test_booleans_do_not_become_number_boxes():
    """`isinstance(True, int)` เป็นจริงใน Python — ถ้าเช็คผิดลำดับ ช่องกาถูกจะกลาย
    เป็นช่องกรอกตัวเลข ซึ่งผู้สอนจะพิมพ์ 1/0 ลงไปแล้วงงว่าทำไมมันแปลก"""
    from dataclasses import dataclass

    @dataclass
    class Sample:
        flag: bool = True
        number: int = 1

    got = {f.key: f.type for f in derive(Sample)}
    assert got["flag"] == "bool"
    assert got["number"] == "int"


def test_a_declared_choice_list_makes_it_an_enum():
    from dataclasses import dataclass

    @dataclass
    class Sample:
        mode: str = "a"

    got = derive(Sample, {"mode": Limit(choices=("a", "b"))})[0]
    assert got.type == "enum" and got.choices == ("a", "b")


def test_config_nested_more_than_one_level_fails_loudly():
    """ฟอร์มที่ตกฟิลด์ไปเงียบๆ แย่กว่าฟอร์มที่สร้างไม่ได้"""
    from dataclasses import dataclass, field

    @dataclass
    class Deep:
        x: int = 1

    @dataclass
    class Mid:
        deep: Deep = field(default_factory=Deep)

    @dataclass
    class Top:
        mid: Mid = field(default_factory=Mid)

    with pytest.raises(TypeError, match="ซ้อนเกินหนึ่งชั้น"):
        derive(Top)


# ── CP463 (agent_env) ───────────────────────────────────────────────

vacuum = pytest.importorskip("vacuum", reason="ต้องติดตั้ง envs/cp463-vacuum")


@pytest.fixture(scope="module")
def vacuum_schema():
    from vacuum.arena import PLUGIN

    return PLUGIN.config_schema()


@pytest.mark.parametrize("phase", ["warmup", "main", "final"])
def test_vacuum_schema_covers_every_key_in_the_shipped_yaml(vacuum_schema, phase):
    """ไฟล์ที่ใช้อยู่จริงคือหลักฐานว่าคีย์ไหนมีจริง — schema ต้องครอบให้ครบ"""
    from vacuum.config import CONFIG_DIR

    missing = yaml_keys(CONFIG_DIR / f"{phase}.yaml") - keys_of(vacuum_schema)
    assert not missing, f"{phase}.yaml มีคีย์ที่ฟอร์มไม่รู้จัก: {sorted(missing)}"


def test_vacuum_enum_choices_match_the_modules_own_constants(vacuum_schema):
    """ตัวเลือกบนฟอร์มต้องมาจากค่าคงที่เดียวกับที่ validate() ใช้ ไม่ใช่พิมพ์ซ้ำ"""
    from vacuum.config import DIRT_DISTRIBUTIONS, OBSTACLE_GENERATORS

    got = {f["key"]: tuple(f.get("choices", ())) for f in vacuum_schema}
    assert got["room.obstacle_generator"] == tuple(OBSTACLE_GENERATORS)
    assert got["room.dirt_distribution"] == tuple(DIRT_DISTRIBUTIONS)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("room.width", 1),                      # ต่ำกว่า minimum
        ("room.obstacle_density", 1.5),         # เกิน maximum
        ("room.dirt_ratio", 0.0),               # ต้องมากกว่า 0
        ("dynamics.action_noise", -0.1),
        ("dynamics.sticky_dirt", 2.0),
        ("room.obstacle_generator", "ไม่มีจริง"),
        ("room.dirt_distribution", "ไม่มีจริง"),
    ],
)
def test_values_outside_the_declared_limits_are_really_rejected(key, bad):
    """**ข้อสำคัญที่สุด** — ขอบเขตที่ประกาศต้องตรงกับที่ `validate()` บังคับจริง

    ถ้าประกาศหลวมกว่าความจริง ฟอร์มจะรับค่าที่ loader ปฏิเสธ · ถ้าประกาศแคบกว่า
    ผู้สอนจะตั้งค่าที่ระบบรองรับไม่ได้ · ทั้งสองแบบไม่มีอะไรฟ้องนอกจากเทสต์นี้
    """
    from vacuum import load_config
    from vacuum.config import CONFIG_DIR, ConfigError

    base = load_config(CONFIG_DIR / "main.yaml")
    with pytest.raises(ConfigError):
        base.replace(**{key: bad})


def test_vacuum_marks_the_untouchable_fields_as_fixed(vacuum_schema):
    """`guarantee_connected: false` ไม่รองรับใน v1.0.0 — ฟอร์มต้องไม่เสนอให้ปิด"""
    fixed = {f["key"] for f in vacuum_schema if f["fixed"]}
    assert {"task", "version", "room.guarantee_connected", "scoring.metric"} <= fixed


# ── CP462 (prediction) ──────────────────────────────────────────────

tabular = pytest.importorskip("tabular", reason="ต้องติดตั้ง envs/cp462-tabular")


@pytest.fixture(scope="module")
def tabular_schema():
    from tabular.arena import PLUGIN

    return PLUGIN.config_schema()


@pytest.fixture
def tabular_base():
    """สเปคที่ใช้ได้จริงหนึ่งอัน — ไม่ต้องมีคลัง เพราะ `replace` ไม่ได้อ่านไฟล์

    เดิมข้อพวกนี้อ่านจาก `configs/churn.yaml` ที่แพ็กไปกับแพ็กเกจ · ไฟล์นั้นไม่มี
    แล้วเพราะ config ของโจทย์อยู่ในฐานข้อมูล และชุดข้อมูลเป็นไฟล์ที่ผู้สอนอัปโหลด
    """
    from tabular.config import TaskSpec

    return TaskSpec(
        title="ทดสอบ", kind="classification", primary="macro_f1",
        dataset="sha256:" + "ab" * 32, target="y",
        split_seed=1, bootstrap_seed=2, labels=[0, 1],
    )


def test_the_form_describes_every_field_of_the_spec(tabular_schema):
    """ฟอร์มต้องครอบทุกฟิลด์ของ `TaskSpec` — ฟิลด์ที่ตกไปจะทำให้ config ไม่ครบ

    `derive()` อนุมานให้อยู่แล้ว ข้อนี้จึงเป็นตัวจับกรณีที่มีใครไปเขียนรายการ
    ฟิลด์ด้วยมือทับในอนาคต
    """
    from tabular.config import TaskSpec

    assert keys_of(tabular_schema) == set(TaskSpec.__dataclass_fields__)


def test_tabular_enum_choices_come_from_the_config_module(tabular_schema):
    from tabular.config import KINDS, PRIMARY_BY_KIND

    got = {f["key"]: tuple(f.get("choices", ())) for f in tabular_schema}
    assert got["kind"] == tuple(KINDS)
    assert set(got["primary"]) == {m for ms in PRIMARY_BY_KIND.values() for m in ms}


def test_no_field_can_be_typed_by_hand_when_it_should_be_picked(tabular_schema):
    """ค่าที่พิมพ์เองไม่ได้ในทางปฏิบัติ ต้องมีตัวควบคุมของตัวเอง

    `dataset` เป็นเลข 64 หลักที่ได้จากการอัปโหลด ส่วน `target` กับ `drop` เป็น
    ชื่อคอลัมน์ที่มีอยู่ในไฟล์นั้นเท่านั้น · ช่องพิมพ์ธรรมดาสำหรับสามอันนี้คือ
    ช่องที่พิมพ์ผิดได้ และความผิดจะไปโผล่ตอนนิสิตส่งงานเข้ามาแล้ว
    """
    widgets = {f["key"]: f.get("widget", "") for f in tabular_schema}
    assert widgets["dataset"] == "upload"
    assert widgets["target"] == "column"
    assert widgets["drop"] == "columns"


def test_the_form_never_asks_for_the_split_the_students_do_themselves(tabular_schema):
    """นิสิตแบ่ง train/val/test เองด้วยเมล็ดของเขา — ฟอร์มต้องไม่มีช่องพวกนั้น

    ถ้ายังมี ผู้สอนจะตั้งค่าที่ระบบไม่ได้ใช้ แล้วเข้าใจว่ามันมีผลกับคะแนน
    """
    assert not ({"ratios", "n_rows", "data_seed", "grading_rows", "grading_seed", "task", "slug"}
                & keys_of(tabular_schema))


@pytest.mark.parametrize(
    "field,bad",
    [
        ("kind", "clustering"),
        ("primary", "roc_auc"),
        ("dataset", "churn"),
        ("student_ratio", 0.05),
        ("student_ratio", 0.99),
        ("final_ratio", 0.9),
    ],
)
def test_tabular_values_outside_the_declared_limits_are_really_rejected(field, bad, tabular_base):
    from tabular.config import ConfigError

    with pytest.raises(ConfigError):
        tabular_base.replace(**{field: bad})


# ── ทะเบียนที่ API ส่งออกไป ─────────────────────────────────────────


def test_the_catalogue_lists_both_environments_with_their_task_type():
    from core.wiring import environments

    got = {e["env_plugin"]: e for e in environments()}
    assert got["vacuum.arena:PLUGIN"]["task_type"] == "agent_env"
    assert got["tabular.arena:PLUGIN"]["task_type"] == "prediction"
    assert all(e["fields"] for e in got.values()), "ทุก env ต้องมีหน้าตาของ config"


def test_every_registered_task_type_has_a_validator():
    """ผูกสองทะเบียนเข้าด้วยกัน — env ที่สร้าง competition ได้แต่ตรวจไฟล์ไม่ได้
    จะรับ submission เข้ามาแล้วล้มตอนตรวจ ซึ่งไม่มีใครสังเกตจนกว่าจะมีคนส่งงาน"""
    from core.wiring import VALIDATORS, environments

    for env in environments():
        assert env["task_type"] in VALIDATORS, f"{env['env_plugin']}: ไม่มีตัวตรวจไฟล์"


# ── ขอบเขตทุกตัวที่ประกาศ ต้องเป็นของจริง (ไล่อัตโนมัติ) ────────────


def out_of_range(field) -> list:
    """ค่าที่ควรถูกปฏิเสธตามขอบเขตที่ประกาศไว้"""
    out = []
    if field.get("choices"):
        out.append("ค่าที่ไม่มีในรายการ")
    if field.get("minimum") is not None:
        out.append(field["minimum"] - 1 if field["type"] == "int" else field["minimum"] - 0.5)
    if field.get("maximum") is not None:
        out.append(field["maximum"] + 1 if field["type"] == "int" else field["maximum"] + 0.5)
    return out


def check_limits_are_real(schema, base, error):
    """ทุกขอบเขตที่ประกาศ ต้องมี loader บังคับอยู่จริง — ไล่จาก schema เอง

    ต่างจากเทสต์รายข้อข้างบนตรงที่ **ฟิลด์ใหม่เข้าข่ายอัตโนมัติ** · ประกาศขอบเขต
    ที่ loader ไม่ได้บังคับ = ฟอร์มปฏิเสธค่าที่ระบบรองรับได้จริง ซึ่งผู้สอนจะเจอ
    เป็น "ตั้งค่านี้ไม่ได้" โดยไม่มีเหตุผล
    """
    unenforced = []
    for field in schema:
        if field["fixed"]:
            continue
        for bad in out_of_range(field):
            try:
                base.replace(**{field["key"]: bad})
            except error:
                continue
            except Exception:
                continue  # ปฏิเสธด้วยชนิดอื่นก็ถือว่าปฏิเสธ
            unenforced.append(f"{field['key']} = {bad!r}")
    return unenforced


def test_every_declared_limit_in_vacuum_is_enforced_by_the_loader(vacuum_schema):
    from vacuum import load_config
    from vacuum.config import CONFIG_DIR, ConfigError

    base = load_config(CONFIG_DIR / "main.yaml")
    unenforced = check_limits_are_real(vacuum_schema, base, ConfigError)
    assert not unenforced, (
        "ขอบเขตที่ประกาศแต่ loader ไม่ได้บังคับ — ฟอร์มจะปฏิเสธค่าที่ระบบรับได้จริง:\n  "
        + "\n  ".join(unenforced)
    )


def test_every_declared_limit_in_tabular_is_enforced_by_the_loader(tabular_schema, tabular_base):
    from tabular.config import ConfigError

    unenforced = check_limits_are_real(tabular_schema, tabular_base, ConfigError)
    assert not unenforced, (
        "ขอบเขตที่ประกาศแต่ loader ไม่ได้บังคับ:\n  " + "\n  ".join(unenforced)
    )


# ── โจทย์ที่แต่ละ env ประกาศว่าเสิร์ฟได้ ───────────────────────────


def offers_of(plugin) -> dict:
    return {o["id"]: o for o in plugin.offers()}


def test_the_three_choices_the_instructor_sees_are_exactly_these():
    """ผู้สอนเลือกจากสามแบบ ไม่ใช่จากสองแกน (`task_type` กับ `kind`)

    ถ้ามีใครเพิ่ม offer ใหม่ เทสต์นี้จะเตือนให้ไปคิดว่าหน้าเว็บกับเอกสารต้องแก้ตามไหม
    """
    from core.wiring import environments

    ids = [o["id"] for e in environments() for o in e["offers"]]
    assert ids == ["reinforcement-learning", "classification", "regression"]


def test_every_offer_hides_the_field_it_already_answers():
    """ผู้สอนที่เลือก Classification ต้องไม่ต้องมากรอก `kind: classification` ซ้ำ"""
    from tabular.arena import PLUGIN

    for oid, offer in offers_of(PLUGIN).items():
        assert offer["defaults"]["kind"] == oid
        assert "kind" in offer["hide"], f"{oid}: ตอบ kind ให้แล้วแต่ยังโชว์ช่องให้กรอก"


def test_narrowed_choices_are_a_subset_of_what_the_field_really_allows():
    """จำกัดตัวเลือกได้ แต่ห้ามเสนอค่าที่ฟิลด์นั้นไม่รองรับตั้งแต่แรก"""
    from tabular.arena import PLUGIN

    fields = {f["key"]: f for f in PLUGIN.config_schema()}
    for oid, offer in offers_of(PLUGIN).items():
        for key, allowed in offer["narrow"].items():
            full = set(fields[key].get("choices") or ())
            assert set(allowed) <= full, f"{oid}.{key}: เสนอค่าที่ไม่มีในฟิลด์จริง"


@pytest.mark.parametrize("oid", ["classification", "regression"])
def test_an_offers_defaults_really_load(oid, tabular_base):
    """**ข้อสำคัญที่สุด** — ค่าที่ตัวเลือกกำหนดให้ ต้องประกอบเป็น config ที่ใช้ได้จริง

    ถ้าไม่ตรง ผู้สอนจะเลือกชนิดโจทย์ กรอกครบ กดบันทึก แล้วโดนปฏิเสธด้วยเหตุผล
    ที่พูดถึงฟิลด์ที่ฟอร์มไม่ได้แสดงให้เห็นด้วยซ้ำ
    """
    from tabular.arena import PLUGIN

    tabular_base.replace(**offers_of(PLUGIN)[oid]["defaults"])   # ต้องไม่โยน


@pytest.mark.parametrize("oid", ["classification", "regression"])
def test_every_narrowed_metric_actually_works_with_that_kind(oid, tabular_base):
    """ทุกคะแนนหลักที่เสนอ ต้องเข้าคู่กับ kind นั้นได้จริง — ไล่จากที่ประกาศเอง"""
    from tabular.arena import PLUGIN

    offer = offers_of(PLUGIN)[oid]
    for metric in offer["narrow"]["primary"]:
        tabular_base.replace(**{**offer["defaults"], "primary": metric})   # ต้องไม่โยน


@pytest.mark.parametrize("oid", ["classification", "regression"])
def test_a_metric_from_the_other_kind_is_still_rejected(oid, tabular_base):
    """พิสูจน์ว่าการจำกัดมีความหมาย — ไม่ใช่ว่าอะไรก็ผ่านอยู่แล้ว"""
    from tabular.arena import PLUGIN
    from tabular.config import ConfigError

    offers = offers_of(PLUGIN)
    other = "regression" if oid == "classification" else "classification"
    with pytest.raises(ConfigError):
        tabular_base.replace(
            **{**offers[oid]["defaults"], "primary": offers[other]["narrow"]["primary"][0]}
        )


def test_the_web_page_builds_its_choices_from_offers_not_from_hardcoded_names():
    """หน้าเว็บต้องไม่รู้จักชื่อโจทย์เอง — เพิ่ม env ที่สามต้องไม่ต้องแก้หน้าเว็บ"""
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent.parent
    index = (repo / "web" / "index.html").read_text(encoding="utf-8")
    assert "e.offers" in index, "หน้าเว็บไม่ได้อ่านรายการ offer จาก API"
    for hardcoded in ('"classification"', '"regression"', '"agent_env"'):
        assert hardcoded not in index, f"หน้าเว็บฝังชื่อ {hardcoded} ไว้ตรงๆ"
