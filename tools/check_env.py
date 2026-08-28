"""ตรวจ /etc/arena.env โดยไม่แสดงค่า secret ออกมา

ปัญหาที่พบบ่อยที่สุดคือช่องว่างและเครื่องหมายคำพูดที่ติดมาโดยไม่ตั้งใจ —
systemd เก็บอัญประกาศไปเป็นส่วนหนึ่งของค่า ทำให้ Google ปฏิเสธโดยไม่บอกสาเหตุ
"""
import pathlib
import sys

REQUIRED = ("ARENA_GOOGLE_CLIENT_ID", "ARENA_GOOGLE_CLIENT_SECRET", "ARENA_WEB_ORIGIN")

#: ไม่บังคับ แต่ถ้าไม่มีจะไม่มีใครเปลี่ยนขนาดทีมจากหน้าเว็บได้เลย
OPTIONAL = ("ARENA_STAFF_EMAILS",)
path = pathlib.Path("/etc/arena.env")

st = path.stat()
import grp, pwd
print(f"ไฟล์   {path}")
print(f"  โหมด {oct(st.st_mode)[-3:]} · เจ้าของ {pwd.getpwuid(st.st_uid).pw_name}:"
      f"{grp.getgrgid(st.st_gid).gr_name} · {st.st_size} ไบต์")
if oct(st.st_mode)[-3:] not in ("600", "640", "400"):
    print("  ⚠️ สิทธิ์กว้างเกินไปสำหรับไฟล์ที่มี secret")

seen, problems = {}, []
for i, line in enumerate(path.read_text().splitlines(), 1):
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    if "=" not in s:
        problems.append(f"บรรทัด {i}: ไม่มีเครื่องหมาย =")
        continue
    k, v = line.split("=", 1)
    seen[k.strip()] = v
    if k != k.strip():
        problems.append(f"{k.strip()}: มีช่องว่างรอบชื่อตัวแปร")
    if v != v.strip():
        problems.append(f"{k.strip()}: มีช่องว่างหน้า/หลังค่า")
    if len(v.strip()) >= 2 and v.strip()[0] == v.strip()[-1] and v.strip()[0] in "\"'":
        problems.append(f"{k.strip()}: ค่าถูกครอบด้วยเครื่องหมายคำพูด — systemd จะเก็บติดไปด้วย")

print("\nค่าที่พบ")
for k in REQUIRED:
    v = seen.get(k)
    if v is None:
        problems.append(f"{k}: ไม่มีในไฟล์")
        continue
    v = v.strip()
    if k.endswith("SECRET"):
        print(f"  {k:<28} ยาว {len(v)} ตัวอักษร · ขึ้นต้นด้วย {v[:6]}…")
        if not v.startswith("GOCSPX-"):
            problems.append(f"{k}: ปกติ client secret ของ Google ขึ้นต้นด้วย GOCSPX-")
    else:
        print(f"  {k:<28} {v}")

for k in OPTIONAL:
    v = (seen.get(k) or "").strip()
    if not v:
        print(f"  {k:<28} (ไม่ได้ตั้ง — ไม่มีใครเปลี่ยนขนาดทีมจากหน้าเว็บได้)")
        continue
    emails = [e.strip() for e in v.split(",") if e.strip()]
    print(f"  {k:<28} {len(emails)} คน · {', '.join(emails)}")
    for e in emails:
        if "@" not in e:
            problems.append(f"{k}: {e!r} ไม่ใช่อีเมล")
        elif e != e.lower():
            # จับคู่ด้วยตัวพิมพ์เล็กอยู่แล้ว แต่การเขียนไม่ตรงกันชวนให้เข้าใจผิดว่าพัง
            print(f"      ℹ️ {e} มีตัวพิมพ์ใหญ่ — ระบบไม่สนใจตัวพิมพ์ ใช้ได้ปกติ")

extra = [k for k in seen if k not in REQUIRED + OPTIONAL]
if extra:
    print(f"  ตัวแปรอื่น: {extra}")

print()
if problems:
    print("✗ ต้องแก้")
    for p in problems:
        print(f"   · {p}")
    sys.exit(1)
print("✓ ไฟล์ถูกต้อง พร้อมให้ systemd อ่าน")
