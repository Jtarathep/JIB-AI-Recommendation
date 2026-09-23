import re
import streamlit as st
from build_engine import parse_requirement, recommend_builds, train_ranker, recommendation_reasons, MODEL_PATH, PRODUCTS, _cpu_matches, _mainboard_quality_score, _mainboard_requirement, _ram_mainboard_fit

st.set_page_config(
    page_title="JIB AI Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentation-first visual layer. This only changes the Streamlit presentation
# and does not alter recommendation / compatibility logic below.
st.markdown("""
<style>
/* ===== Modern Presentation UI ===== */
#MainMenu, footer {visibility:hidden;}
header {background:transparent !important;}
.block-container {max-width:1280px; padding-top:1.25rem; padding-bottom:2.5rem;}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0a1020 55%, #070b15 100%);
    border-right:1px solid rgba(94,126,255,.18);
}
[data-testid="stSidebar"] > div:first-child {padding-top:1.15rem;}

/* Hero */
.kk-hero {
    position:relative; overflow:hidden;
    padding:1.55rem 1.65rem 1.35rem;
    border:1px solid rgba(105,129,255,.34);
    border-radius:26px;
    background:
      radial-gradient(circle at 86% 18%, rgba(119,72,255,.30), transparent 28%),
      radial-gradient(circle at 70% 92%, rgba(0,207,255,.16), transparent 30%),
      linear-gradient(135deg,#081a30 0%,#101636 52%,#11102a 100%);
    box-shadow:0 20px 60px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.06);
    margin-bottom:1rem;
}
.kk-hero:before,.kk-hero:after {content:""; position:absolute; border-radius:999px; filter:blur(2px); pointer-events:none;}
.kk-hero:before {width:280px;height:280px;right:-120px;top:-150px;background:rgba(91,74,255,.18);}
.kk-hero:after {width:210px;height:210px;left:-130px;bottom:-150px;background:rgba(0,210,255,.10);}
.kk-eyebrow {display:inline-flex;align-items:center;gap:.45rem;color:#77b9ff;font-size:.74rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.42rem;}
.kk-title {font-size:clamp(2rem,4.2vw,3.55rem);font-weight:900;letter-spacing:-.045em;line-height:1.02;margin:0;position:relative;
    background:linear-gradient(90deg,#fff 0%,#8bdcff 43%,#9d8cff 78%,#ef9cff 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.kk-subtitle {margin-top:.62rem;color:#d6def1;font-size:1rem;line-height:1.62;max-width:930px;position:relative;}
.kk-subtitle strong {color:#fff;}

/* Feature cards */
.kk-badges {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;margin-top:1.15rem;position:relative;}
.kk-badge {min-height:86px;padding:.85rem .95rem;border-radius:17px;border:1px solid rgba(134,160,255,.22);
    background:linear-gradient(145deg,rgba(29,50,93,.78),rgba(13,22,42,.88));
    color:#f5f8ff;font-size:.88rem;font-weight:800;display:flex;flex-direction:column;justify-content:center;gap:.28rem;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 8px 24px rgba(0,0,0,.12);}
.kk-badge:nth-child(1){border-color:rgba(218,67,255,.45);box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 0 22px rgba(190,50,255,.09);}
.kk-badge:nth-child(2){border-color:rgba(0,223,183,.35);}
.kk-badge:nth-child(3){border-color:rgba(65,148,255,.42);}
.kk-badge:nth-child(4){border-color:rgba(255,188,67,.35);}
.kk-badge span{color:#aebddb;font-size:.72rem;font-weight:500;line-height:1.35;}

/* Process */
.kk-flow {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:.85rem 0 1rem;padding:.55rem;border:1px solid rgba(99,125,213,.20);border-radius:19px;background:rgba(10,20,39,.72);}
.kk-step {position:relative;padding:.78rem .75rem;border-radius:14px;background:linear-gradient(145deg,rgba(27,43,78,.8),rgba(14,22,43,.8));border:1px solid rgba(110,139,229,.16);text-align:center;font-size:.8rem;color:#aebbd4;}
.kk-step strong{display:block;color:#f4f7ff;font-size:.9rem;margin-bottom:.18rem;}
.kk-step:first-child{border-color:rgba(93,129,255,.35);}.kk-step:nth-child(2){border-color:rgba(0,218,198,.30);}.kk-step:nth-child(3){border-color:rgba(72,135,255,.30);}.kk-step:nth-child(4){border-color:rgba(160,103,255,.32);}

.kk-note {border:1px solid rgba(0,210,190,.24);border-left:4px solid #27d6bf;padding:.82rem 1rem;margin:.55rem 0 1.15rem;background:linear-gradient(90deg,rgba(0,191,174,.10),rgba(72,89,255,.07));border-radius:0 14px 14px 0;color:#aebbd4;}
.kk-note b{color:#f5f8ff;}

/* Chat area */
[data-testid="stChatMessage"] {border-radius:18px;}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {font-size:1rem;line-height:1.65;}
[data-testid="stChatInput"] {border-radius:18px !important;}

/* Buttons */
.stButton > button {border-radius:12px !important;border:1px solid rgba(111,137,230,.20) !important;transition:.18s ease !important;}
.stButton > button:hover {border-color:rgba(126,155,255,.55) !important;transform:translateY(-1px);box-shadow:0 8px 22px rgba(55,80,180,.15);}

/* Sidebar content */
.kk-side-brand {padding:.5rem .15rem 1rem;}
.kk-side-brand .brand {font-size:1.25rem;font-weight:900;letter-spacing:-.02em;color:#fff;}
.kk-side-brand .line {height:2px;width:42px;background:linear-gradient(90deg,#23d9ff,#9d6cff);border-radius:10px;margin-top:.35rem;}
.kk-side-card {padding:.9rem;border:1px solid rgba(105,132,220,.18);border-radius:16px;background:linear-gradient(145deg,rgba(20,35,66,.65),rgba(12,19,35,.7));margin:.55rem 0;}
.kk-side-label {font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:#8093b8;font-weight:800;}

@media (max-width: 900px){.kk-badges{grid-template-columns:repeat(2,minmax(0,1fr));}.kk-flow{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width: 600px){.block-container{padding-top:.7rem}.kk-hero{padding:1.15rem}.kk-badges{grid-template-columns:1fr}.kk-flow{grid-template-columns:1fr}.kk-title{font-size:2rem}.kk-subtitle{font-size:.9rem}}
</style>

<div class="kk-hero">
  <div class="kk-eyebrow">✦ PROTOTYPE PROJECT</div>
  <div class="kk-title">🤖 JIB AI Recommendation System</div>
  <div class="kk-subtitle"><strong>ระบบแนะนำชุดคอมพิวเตอร์ด้วย AI</strong> พร้อม Machine Learning และการตรวจสอบความเข้ากันได้ของอุปกรณ์ สำหรับกรณีศึกษา JIB</div>
  <div class="kk-badges">
    <div class="kk-badge">🧠 AI Recommendation<span>วิเคราะห์ความต้องการและแนะนำชุดคอมที่เหมาะสม</span></div>
    <div class="kk-badge">⚙️ Machine Learning<span>ใช้โมเดล ML ช่วยจัดอันดับผลลัพธ์</span></div>
    <div class="kk-badge">🛡️ PC Compatibility<span>ตรวจสอบความเข้ากันได้ของอุปกรณ์ทุกชิ้น</span></div>
    <div class="kk-badge">📊 JIB Case Study<span>กรณีศึกษาการแนะนำสินค้า PC</span></div>
  </div>
</div>

<div class="kk-flow">
  <div class="kk-step"><strong>① Requirement</strong>งบประมาณ + การใช้งาน</div>
  <div class="kk-step"><strong>② AI Analysis</strong>วิเคราะห์และจัดอันดับ</div>
  <div class="kk-step"><strong>③ PC Builds</strong>Build 1 / 2 / 3</div>
  <div class="kk-step"><strong>④ Compatibility</strong>ตรวจสอบความเข้ากันได้</div>
</div>

<div class="kk-note">💡 <b>Demo</b> · ลองพิมพ์ <b>งบ 40,000 Gaming</b> หรือ <b>งบ 40,000 Programming</b> เพื่อเริ่มการแนะนำชุดคอมพิวเตอร์</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    with st.spinner("กำลังเตรียม ML model สำหรับ Prototype..."):
        train_ranker()

WELCOME = "สวัสดีครับ บอกงบประมาณและลักษณะการใช้งานได้เลย เช่น งบ 30,000 เล่นเกม + เขียนโปรแกรม RAM 32GB DDR5 SSD 1TB"
USAGE_OPTIONS = [
    ("🎮 Gaming", "gaming", "Gaming"),
    ("💼 ใช้งานทั่วไป", "general", "ใช้งานทั่วไป / Office"),
    ("💻 เขียนโปรแกรม", "programming", "เขียนโปรแกรม / Development"),
    ("🎨 ตัดต่อ / กราฟิก", "editing", "ตัดต่อ / กราฟิก"),
    ("🧠 AI / ML", "ai_ml", "AI / Machine Learning"),
]
USAGE_KEYS = {x[1] for x in USAGE_OPTIONS}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
if "requirements" not in st.session_state:
    st.session_state.requirements = {}
if "pending_edit" not in st.session_state:
    st.session_state.pending_edit = None
if "current_builds" not in st.session_state:
    st.session_state.current_builds = []


def usage_labels(req):
    return [label for _, key, label in USAGE_OPTIONS if req.get(key)]


def clear_usage(req):
    for key in USAGE_KEYS:
        req.pop(key, None)


def has_new_budget(text):
    """Detect a genuinely new budget request.

    A bare number can be a budget (e.g. "40000 Gaming"), but values such as
    "32GB" after an edit request must not reset the whole conversation.
    """
    t = text.lower().replace(",", "").strip()
    if re.search(r"(?:งบ(?:ประมาณ)?\s*)\d+(?:\.\d+)?\s*(?:บาท|฿|k\b)?", t):
        return True
    # Explicit currency / k budget without the Thai "งบ" prefix.
    if re.fullmatch(r"\d+(?:\.\d+)?\s*(?:บาท|฿|k)", t):
        return True
    # Bare numeric budget, but never a storage/RAM unit.
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return True
    # Mixed requests such as "40000 Gaming".
    if re.match(r"^\d+(?:\.\d+)?\s*(?:k\b)?\s+(?:gaming|programming|coding|general|editing|ai|เล่นเกม|เขียนโปรแกรม|ใช้งานทั่วไป|ตัดต่อ)", t):
        return True
    return False




def budget_question_kind(text):
    """Detect natural-language questions about the minimum/maximum budget.

    This is intentionally a lightweight intent parser; it does not replace the
    recommendation engine.
    """
    t = str(text or "").strip().lower().replace(",", "")
    min_words = [
        "งบต่ำสุด", "งบประมาณต่ำสุด", "ถูกที่สุด", "ถูกสุด",
        "ถูกสุดเท่าไหร่", "งบน้อยที่สุด", "งบน้อยสุด",
        "คอมถูกที่สุด", "คอมถูกสุด", "จัดคอมถูก",
        "minimum budget", "lowest budget", "cheapest pc", "cheapest computer",
    ]
    max_words = [
        "งบสูงสุด", "งบประมาณสูงสุด", "แพงที่สุด", "แพงสุด",
        "งบมากที่สุด", "งบมากสุด", "คอมแพงที่สุด", "คอมแพงสุด",
        "จัดคอมแพง", "maximum budget", "highest budget", "most expensive pc",
    ]
    if any(x in t for x in min_words):
        return "min"
    if any(x in t for x in max_words):
        return "max"
    return None


def estimate_min_budget():
    """Find the cheapest complete general-purpose PC the current prototype can build.

    A minimum-budget question has no usage specified, so use the prototype's
    General/Office baseline rather than a budget-only request. The returned
    number is the actual price of the cheapest feasible build found, not merely
    the first input-budget threshold at which the recommender happens to return.
    """
    from build_engine import recommend_builds

    low, high = 10000, 60000
    first = None
    for budget in range(low, high + 1, 500):
        builds = recommend_builds({"budget": budget, "general": True}, top_n=3)
        if builds:
            first = budget
            break
    if first is None:
        return None

    # Refine the feasibility threshold.
    lo = max(low, first - 499)
    hi = first
    while hi - lo > 20:
        mid = (lo + hi) // 2
        builds = recommend_builds({"budget": mid, "general": True}, top_n=3)
        if builds:
            hi = mid
        else:
            lo = mid + 1

    # Use the actual cheapest feasible build, not the requested threshold.
    builds = recommend_builds({"budget": hi, "general": True}, top_n=3)
    if not builds:
        return None
    build = min(builds, key=lambda x: x.get("total", float("inf")))
    return {"budget": int(hi), "total": int(round(build["total"])), "build": build}


def estimate_high_budget_reference():
    """Find a high-price compatible build from the current Dataset.

    The previous version used 100,000 as the search budget. That made the
    answer look like a maximum even when the Dataset could still produce a
    compatible build above 100,000 (for example when the user later entered
    a 200,000 budget). Use a high enough ceiling for this Dataset, then report
    the highest-priced compatible build returned by the recommender.
    """
    from build_engine import recommend_builds
    search_budget = 200000
    # No usage is required for this question. Gaming is used only as a
    # high-performance baseline so the reference does not accidentally favor
    # a low-power office configuration.
    builds = recommend_builds({"budget": search_budget, "gaming": True}, top_n=3)
    if not builds:
        return None
    best = max(builds, key=lambda x: x.get("total", 0))
    return {"total": int(round(best["total"])), "build": best, "search_budget": search_budget}


def budget_question_reply(kind):
    if kind == "min":
        result = estimate_min_budget()
        if not result:
            return ("ตอนนี้ระบบยังหาชุด PC ขั้นต่ำจาก Dataset ปัจจุบันไม่ได้ครับ "
                    "จึงไม่ควรเดาตัวเลขงบขั้นต่ำขึ้นมาเอง")
        return (
            f"### 💰 งบประมาณขั้นต่ำที่ระบบจัดชุดได้\n\n"
            f"ประมาณ **{result['total']:,} บาท**\n\n"
            "ตัวเลขนี้เป็นงบขั้นต่ำโดยประมาณจาก Dataset และกฎ Compatibility "
            "ของ Prototype ในปัจจุบัน ไม่ได้หมายความว่าเป็นราคาต่ำสุดของตลาดทั้งหมด\n\n"
            "ถ้าต้องการ ผมสามารถนำงบนี้ไปจัด Build 1 / Build 2 / Build 3 "
            "ตามการใช้งานที่คุณต้องการได้ครับ"
        )
    result = estimate_high_budget_reference()
    if not result:
        return ("ตอนนี้ระบบยังสร้างชุดงบสูงจาก Dataset ปัจจุบันไม่ได้ครับ "
                "จึงไม่ควรเดาตัวเลขขึ้นมาเอง")
    return (
        f"### 💰 งบประมาณสูงสุดที่ระบบจัดชุดอ้างอิงได้\n\n"
        f"ประมาณ **{result['total']:,} บาท**\n\n"
        f"ตัวเลขนี้มาจาก Build ที่มีราคาสูงสุดที่ระบบค้นพบและผ่าน Compatibility "
        f"ภายใน Dataset ปัจจุบัน โดยค้นด้วยเพดานงบ **{result.get('search_budget', 200000):,} บาท** "
        "ไม่ใช่การรับประกันว่าเป็นราคาสูงสุดของสินค้าทั้งตลาด\n\n"
        "หากต้องการจัดชุดจริง ให้ระบุการใช้งาน เช่น **Gaming / Programming / Editing / AI** "
        "เพราะงบสูงสุดที่เหมาะสมจะแตกต่างกันตามงานครับ"
    )


def is_ram_edit_request(text):
    """Recognize the conversational command 'เพิ่ม RAM' without changing the
    existing budget/usage context yet."""
    t = str(text or "").strip().lower()
    return bool(re.search(r"\b(?:เพิ่ม|อัป|อัพ|upgrade)\s*(?:ram|แรม)\b", t))


def has_explicit_ram_value(text):
    t = str(text or "").strip().lower()
    return bool(re.search(r"(?:ram|แรม)?\s*(?:ขอ|เป็น|ให้เป็น|เอา)?\s*\d+\s*gb\b", t))


def extract_ram_value(text):
    m = re.search(r"(?:ram|แรม)?\s*(?:ขอ|เป็น|ให้เป็น|เอา)?\s*(\d+)\s*gb\b", str(text or "").strip().lower())
    return int(m.group(1)) if m else None




def _component_edit_command(text, component):
    t = str(text or "").strip().lower()
    if component == "cpu":
        return bool(re.search(r"(?:เปลี่ยน|เปลี่ยนเป็น|สลับ|อัปเกรด|upgrade|change)\s*(?:cpu|ซีพียู|processor)", t))
    if component == "gpu":
        return bool(re.search(r"(?:เปลี่ยน|เปลี่ยนเป็น|สลับ|อัปเกรด|upgrade|change)\s*(?:gpu|การ์ดจอ|การ์ดแสดงผล|graphics card)", t))
    if component == "ram":
        return bool(re.search(r"(?:เปลี่ยน|เปลี่ยนเป็น|เพิ่ม|อัป|อัพ|upgrade|change)\s*(?:ram|แรม)", t))
    if component == "storage":
        return bool(re.search(r"(?:เปลี่ยน|เปลี่ยนเป็น|เพิ่ม|อัป|อัพ|upgrade|change|เพิ่มพื้นที่)\s*(?:ssd|storage|พื้นที่|m\.2|nvme)", t))
    return False


def _has_explicit_gpu(text):
    return bool(re.search(r"(?:rtx\s*\d{3,4}(?:\s*(?:ti|super))?|rx\s*\d{3,4}(?:\s*(?:xt|xtx))?)", str(text or "").lower()))


def _find_cpu_from_text(text):
    t = re.sub(r"\s+", " ", str(text or "").strip().lower())
    for cpu in PRODUCTS.get("cpu", []):
        if _cpu_matches(t, cpu.get("name", "")):
            return cpu
    return None


def _component_label(component):
    return {"cpu": "CPU", "gpu": "GPU", "ram": "RAM", "storage": "SSD / Storage"}.get(component, component)


def _current_builds(req):
    builds = st.session_state.get("current_builds") or []
    if builds:
        return builds
    builds = recommend_builds(req, top_n=3)
    st.session_state.current_builds = builds
    return builds


def _locked_probe(req, base_build, component):
    probe = dict(req)
    b = base_build["build"] if "build" in base_build else base_build
    # Preserve the meaningful parts of the selected Build, but allow the engine
    # to re-select platform-dependent parts when a component changes.
    # CPU -> Mainboard/Cooling/PSU may need to change.
    # GPU -> Mainboard/PSU/Case may need to change.
    # RAM -> Mainboard/PSU may need to change.
    # Storage -> platform can remain stable.
    keep = {
        "cpu": ["gpu", "ram", "storage", "case"],
        "gpu": ["cpu", "ram", "storage", "case"],
        "ram": ["cpu", "gpu", "storage", "case"],
        "storage": ["cpu", "gpu", "ram", "case"],
    }.get(component, [])
    probe["_locked_build"] = {k: b[k] for k in keep if k in b}
    probe["_edit_component"] = component
    return probe


def _ai_component_suggestions(req, component, target_index=None):
    builds = _current_builds(req)
    if target_index is None or target_index == "all":
        # For all-build mode, produce up to three candidates per Build. The UI
        # will show each Build separately so a single model is never silently
        # applied to all alternatives.
        targets = range(len(builds))
    else:
        targets = [target_index]
    result = {}
    for idx in targets:
        if idx >= len(builds):
            continue
        probe = _locked_probe(req, builds[idx], component)
        if component == "cpu":
            probe["focus_cpu"] = True
        elif component == "gpu":
            probe["focus_gpu"] = True
        elif component == "ram":
            probe["focus_ram"] = True
        elif component == "storage":
            probe["focus_ssd"] = True
        found = recommend_builds(probe, top_n=6)
        out=[]; seen=set()
        current_name = builds[idx]["build"][component].get("name", "")
        for r in found:
            item=r["build"][component]
            name=item.get("name","")
            # Candidate list should contain actual alternatives. The current
            # component is shown separately by the UI, not offered as a
            # misleading "change" option.
            if name and name != current_name and name not in seen:
                seen.add(name); out.append((item,r))
        result[idx]=out[:3]
    return result


def _candidate_probe(req, base_build, component, item_name):
    probe = _locked_probe(req, base_build, component)
    if component == "cpu":
        probe["cpu_model"] = item_name
    elif component == "gpu":
        # exact GPU matching is handled by the engine's GPU requirement.
        probe["gpu"] = item_name
    elif component == "ram":
        item = next((x for x in PRODUCTS.get("ram", []) if x.get("name") == item_name), None)
        if item:
            probe["ram_gb"] = item.get("capacity")
            probe["ram_type"] = item.get("ddr")
    elif component == "storage":
        item = next((x for x in PRODUCTS.get("storage", []) if x.get("name") == item_name), None)
        if item:
            probe["storage_gb"] = item.get("capacity")
    return probe


def _apply_component_to_build(req, base_build, component, item_name):
    probe = _candidate_probe(req, base_build, component, item_name)
    result = recommend_builds(probe, top_n=1)
    if not result:
        return None
    return result[0]


def _apply_exact_component_change(req, component, text, target_indices=None):
    builds = _current_builds(req)
    if target_indices is None:
        target_indices = list(range(len(builds)))
    updated = list(builds)
    # Resolve requested product first.
    if component == "cpu":
        item = _find_cpu_from_text(text)
        if not item:
            return False, "ยังหา CPU รุ่นนี้ใน Dataset ปัจจุบันไม่พบครับ ลองพิมพ์ชื่อรุ่น เช่น **i7-14700** หรือ **Ryzen 7 7800X3D**"
        name=item["name"]
    elif component == "gpu":
        parsed = parse_requirement(text, {})
        if not parsed.get("gpu"):
            return False, "ยังหา GPU รุ่นนี้ไม่พบครับ ลองพิมพ์ชื่อรุ่น เช่น **RTX 5070** หรือ **RX 7700 XT**"
        name=parsed["gpu"]
    else:
        return False, ""
    failed=[]
    for idx in target_indices:
        if idx >= len(updated):
            continue
        nr=_apply_component_to_build(req, updated[idx], component, name)
        if nr is None:
            failed.append(idx+1)
        else:
            updated[idx]=nr
    if failed and len(failed)==len(target_indices):
        return False, f"ยังไม่สามารถเปลี่ยน {_component_label(component)} เป็น **{name}** ใน Build ที่เลือกได้ โดยยังคงเงื่อนไขเดิมทั้งหมดไว้ครับ"
    st.session_state.current_builds=updated
    suffix = f"Build {', '.join(str(i+1) for i in target_indices if i+1 not in failed)}"
    if len(target_indices) == len(builds): suffix = "ทุก Build ที่ผ่านเงื่อนไข"
    msg=f"เปลี่ยน {_component_label(component)} เป็น **{name}** สำหรับ {suffix} แล้วครับ ระบบตรวจ Compatibility, Mainboard, RAM, Cooling, PSU และงบใหม่ทั้งหมด"
    if failed:
        msg += f"\n\nBuild ที่ไม่สามารถเปลี่ยนได้โดยยังคงเงื่อนไขเดิม: **{', '.join('Build '+str(i) for i in failed)}**"
    return True,msg


def _reset_current_builds():
    st.session_state.current_builds=[]


def make_reply(req):
    summary = []
    if req.get("budget"):
        summary.append(f"งบประมาณ **{req['budget']:,} บาท**")
    usage = usage_labels(req)
    if usage:
        summary.append("การใช้งาน: **" + " + ".join(usage) + "**")
    if req.get("cpu_model"): summary.append(f"CPU: **{req['cpu_model']}**")
    if req.get("gpu"): summary.append(f"GPU: **{req['gpu']}**")
    if req.get("ram_gb"): summary.append(f"RAM: **{req['ram_gb']}GB**")
    if req.get("ram_type"): summary.append(f"RAM Type: **{req['ram_type']}**")
    if req.get("storage_gb"): summary.append(f"Storage: **{req['storage_gb']}GB**")
    if req.get("wifi"): summary.append("ต้องการ **Wi-Fi**")
    focus = []
    for key, label in [("focus_cpu", "CPU"), ("focus_gpu", "GPU"), ("focus_ram", "RAM"), ("focus_ssd", "SSD"), ("focus_hdd", "HDD")]:
        if req.get(key): focus.append(label)
    if focus: summary.append("เน้น **" + " + ".join(focus) + "**")

    reply = "### Requirement ปัจจุบัน\n" + ("\n".join(f"- {x}" for x in summary) if summary else "- ยังไม่ได้ระบุ Requirement สำคัญ")

    # Do not recommend a build until the usage is known. Budget + focus alone
    # is a new request that must ask for usage first.
    if not usage:
        reply += ("\n\nรับ Requirement แล้วครับ แต่ยังไม่ได้ระบุลักษณะการใช้งาน\n\n"
                  "กรุณาเลือกว่าจะนำคอมไปใช้ทำอะไร แล้วระบบจะจัดชุดใหม่ให้ตามงบและจุดที่เน้นครับ")
        return reply

    builds = _current_builds(req)

    if builds:
        reply += "\n\n### ผลการจัดอันดับชุด PC\n"
        for i, r in enumerate(builds, 1):
            label = "เหมาะสมที่สุดตาม Requirement" if i == 1 else "ทางเลือก"
            reply += f"\n**Build {i} — {label}**\n"
            reply += f"- ราคาโดยรวม: **{r['total']:,.0f} บาท**\n"
            # Score is relative to the three returned recommendations so the
            # UI no longer turns every valid build into 100/100. The rank remains
            # driven by the full rule + ML score in build_engine.py.
            result_scores = [x["score"] for x in builds]
            s_min, s_max = min(result_scores), max(result_scores)
            if s_max > s_min:
                display_score = 80.0 + 15.0 * ((r["score"] - s_min) / (s_max - s_min))
            else:
                display_score = 90.0
            reply += f"- Suitability Score: **{display_score:.1f}/100**\n"
            reply += f"- Compatibility: **{'ผ่าน' if r['compatible'] else 'ไม่ผ่าน'}**\n"
            reply += f"- CPU/GPU Balance: **{r.get('balance_note', 'ประเมินแล้ว')}**\n"
            reply += f"- Cooling: **{r.get('cooling_note', 'ประเมินแล้ว')}**\n"
            cb = r['build']['cpu']; gb = r['build']['gpu']; pb = r['build']['psu']; mb = r['build']['mainboard']
            mb_min, _ = _mainboard_requirement(cb, gb, r['build'].get('ram'))
            mb_score = _mainboard_quality_score(mb)
            reply += f"- Mainboard Fit: **ระดับ {mb_score:.0f}/100 — ขั้นต่ำที่เหมาะกับ CPU ชุดนี้ {mb_min:.0f}/100**\n"
            ram_ok, ram_notes, _ = _ram_mainboard_fit(cb, mb, r['build']['ram'])
            rb = r['build']['ram']
            ram_status = "ผ่าน" if ram_ok else "ไม่ผ่าน"
            reply += f"- RAM Compatibility: **{ram_status} — {rb.get('ddr','')} {rb.get('bus',0):.0f}MHz | Mainboard สูงสุด {mb.get('ram_max_speed',0):.0f}MHz | Profile {mb.get('ram_profile','ไม่ระบุ')}**\n"
            if ram_notes:
                reply += f"- RAM Analysis: **{' / '.join(ram_notes[:2])}**\n"
            estimated_load = r.get('estimated_load', r.get('estimated_power', 0) / 1.30)
            headroom = r.get('psu_headroom', pb['watt'] - estimated_load)
            required_psu = r.get('estimated_power', 0)
            reply += f"- Power Analysis: **Estimated Load ~{estimated_load:.0f}W / PSU แนะนำ {pb['watt']:.0f}W / Headroom +{headroom:.0f}W**\n"
            reply += f"- PSU Target: **ระบบคำนวณขั้นต่ำประมาณ {required_psu:.0f}W แล้วเลือก PSU ที่มีสำรองและไม่กินงบเกินจำเป็น**\n"
            reply += f"- Storage Strategy: **{r['build']['storage'].get('type', 'ไม่ระบุ')} {r['build']['storage'].get('capacity', 0):.0f}GB**\n"
            why = recommendation_reasons(r, req)
            if why:
                reply += "- เหตุผลที่ AI แนะนำ: " + "; ".join(why) + "\n"
            if r.get('gpu_note'):
                reply += f"- GPU: **ตัวเลือกทดแทน** — {r['gpu_note']}\n"
            if r['issues']:
                reply += "- ประเด็น: " + "; ".join(r['issues']) + "\n"
            for key, label2 in [("cpu","CPU"),("mainboard","Mainboard"),("gpu","GPU"),("ram","RAM"),("storage","Storage"),("psu","PSU"),("case","Case"),("cooler","CPU Cooler")]:
                item = r['build'][key]
                reply += f"- {label2}: {item['name']} — {item['price']:,.0f} บาท"
                if item.get('url'):
                    reply += f" — [ดูสินค้า]({item['url']})"
                reply += "\n"
    else:
        reply += "\n\nยังสร้างชุดที่ผ่านเงื่อนไขจาก Dataset ปัจจุบันไม่ได้ครับ ระบบจะไม่สรุปว่าต้องเพิ่มงบโดยอัตโนมัติ แต่จะถือว่าเป็นข้อจำกัดของข้อมูล/เงื่อนไขที่กำหนด"
    return reply



def show_component_edit_controls():
    pending = st.session_state.pending_edit or ""
    valid_prefixes = ("select_build_", "method_", "candidates_", "confirm_", "manual_")
    if not pending.startswith(valid_prefixes):
        return

    req = st.session_state.requirements
    builds = _current_builds(req)
    comp = pending.split("_", 2)[-1] if pending.startswith("select_build_") else st.session_state.get("edit_component")
    if not comp or not builds:
        return

    label = _component_label(comp)
    st.markdown(f"### 🔧 เปลี่ยน {label}")

    # Step 1: select one or more Builds. Do not immediately advance when a
    # Build button is pressed; this lets the user select Build 1 + Build 3,
    # or all three, and confirm the target set once.
    if pending.startswith("select_build_"):
        st.caption("เลือกได้มากกว่า 1 Build แล้วกด 'ดำเนินการต่อ' เมื่อเลือกครบแล้ว")
        selected = set(st.session_state.get("edit_targets", []))
        cols = st.columns(min(4, len(builds)))
        for i in range(len(builds)):
            active = i in selected
            if cols[i].button(
                ("✓ " if active else "") + f"Build {i+1}",
                use_container_width=True,
                key=f"edit_target_{comp}_{i}_{'on' if active else 'off'}"
            ):
                if i in selected:
                    selected.remove(i)
                else:
                    selected.add(i)
                st.session_state.edit_targets = sorted(selected)
                st.rerun()

        if st.button("ทุก Build", use_container_width=True, key=f"edit_target_{comp}_all"):
            st.session_state.edit_targets = list(range(len(builds)))
            st.rerun()

        selected = st.session_state.get("edit_targets", [])
        if selected:
            names = ", ".join(f"Build {i+1}" for i in selected)
            st.success(f"เลือกแล้ว: **{names}**")
        else:
            st.info("ยังไม่ได้เลือก Build")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("ดำเนินการต่อ", disabled=not selected, use_container_width=True, key=f"edit_target_next_{comp}"):
                st.session_state.pending_edit = f"method_{comp}"
                st.rerun()
        with c2:
            if st.button("ยกเลิก", use_container_width=True, key=f"edit_cancel_target_{comp}"):
                st.session_state.pending_edit = None
                st.session_state.edit_targets = []
                st.rerun()
        return

    targets = st.session_state.get("edit_targets", [])
    target_text = "ทุก Build" if len(targets) == len(builds) else ", ".join(f"Build {i+1}" for i in targets)
    st.info(f"กำลังแก้ **{target_text}**")

    # Step 2: choose AI or manual.
    if pending.startswith("method_"):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🤖 ให้ AI เลือก", use_container_width=True, key=f"method_ai_{comp}"):
                st.session_state.edit_selected_candidates = {}
                st.session_state.pending_edit = f"candidates_{comp}"
                st.rerun()
        with c2:
            if st.button("🔍 ระบุรุ่นเอง", use_container_width=True, key=f"method_manual_{comp}"):
                st.session_state.manual_preview = {}
                st.session_state.pending_edit = f"manual_{comp}"
                st.rerun()
        if st.button("ย้อนกลับ", use_container_width=True, key=f"edit_back_method_{comp}"):
            st.session_state.pending_edit = f"select_build_{comp}"
            st.rerun()
        return

    # Step 3A: AI candidates. Each selected Build gets its own candidate and
    # the user may select candidates for several Builds before one confirmation.
    if pending.startswith("candidates_"):
        selected_candidates = st.session_state.get("edit_selected_candidates", {})
        suggestions = _ai_component_suggestions(req, comp, "all" if len(targets) > 1 else targets[0])

        # Always render every selected Build and always render exit controls.
        # A single Build with no valid candidate must not leave the user stuck
        # on this screen.
        any_options = False
        for idx in targets:
            options = suggestions.get(idx, []) if suggestions else []
            any_options = any_options or bool(options)
            st.markdown(f"**Build {idx+1} — AI Candidate**")
            current_name = builds[idx]['build'][comp].get('name', '')
            if current_name:
                st.caption(f"กำลังใช้อยู่: **{current_name}**")

            if not options:
                st.warning(
                    f"❌ ไม่พบ {_component_label(comp)} ที่เหมาะสมสำหรับ Build {idx+1} "
                    "โดยยังคงเงื่อนไขเดิมของชุดนี้ได้ จึง **ยังไม่มีการเปลี่ยนแปลง**"
                )
                st.caption("สามารถย้อนกลับไปเลือกวิธีอื่น / ระบุรุ่นเอง / ยกเลิกได้")
                continue

            cols = st.columns(len(options))
            for j, (item, r) in enumerate(options):
                with cols[j]:
                    current = selected_candidates.get(idx)
                    is_selected = current == item['name']
                    is_current = item['name'] == current_name
                    if is_selected:
                        st.success(f"✓ **{j+1}. {item['name']}**")
                    else:
                        st.markdown(f"**{j+1}. {item['name']}**")
                    st.caption(f"{item['price']:,.0f} บาท | ชุดหลังปรับประมาณ {r['total']:,.0f} บาท")
                    if is_current and not is_selected:
                        st.caption("ใช้อยู่แล้วใน Build นี้")
                        st.button("ใช้อยู่แล้ว", disabled=True, key=f"candidate_current_{comp}_{idx}_{j}", use_container_width=True)
                    else:
                        if st.button("เลือก" if not is_selected else "เลือกแล้ว", key=f"candidate_{comp}_{idx}_{j}", use_container_width=True):
                            selected_candidates[idx] = item['name']
                            st.session_state.edit_selected_candidates = selected_candidates
                            st.rerun()

        chosen = [i for i in targets if i in selected_candidates]
        if chosen:
            st.success("เลือกแล้ว: " + ", ".join(f"Build {i+1}" for i in chosen))
        elif any_options:
            st.info("เลือก Candidate อย่างน้อย 1 Build เพื่อดู Preview ได้เลย ไม่จำเป็นต้องเลือกครบทุก Build")
        else:
            st.error(
                "⚠️ ไม่มี Candidate ที่ผ่านเงื่อนไขสำหรับ Build ที่เลือกเลย "
                "ระบบจะยังไม่เปลี่ยนอะไรจนกว่าจะมีตัวเลือกที่ตรวจสอบผ่าน"
            )

        # These controls must exist even when there is no candidate, or when
        # only one Build was selected.
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(
                "🔍 ตรวจสอบและดู Preview",
                disabled=(not chosen),
                use_container_width=True,
                key=f"confirm_candidates_{comp}"
            ):
                preview = {}
                failed = []
                unchanged = []
                for idx in chosen:
                    name = selected_candidates[idx]
                    old_item = builds[idx]['build'][comp]
                    if old_item.get('name') == name:
                        unchanged.append(idx)
                        continue
                    nr = _apply_component_to_build(req, builds[idx], comp, name)
                    if nr:
                        preview[idx] = nr
                    else:
                        failed.append(idx)
                st.session_state.edit_preview = preview
                st.session_state.edit_preview_failed = failed
                st.session_state.edit_preview_unchanged = unchanged
                st.session_state.edit_preview_unselected = [i for i in targets if i not in chosen]
                st.session_state.pending_edit = f"confirm_{comp}"
                st.rerun()
        with c2:
            if st.button("🔙 เปลี่ยนวิธี / ย้อนกลับ", use_container_width=True, key=f"edit_back_candidates_{comp}"):
                st.session_state.pending_edit = f"method_{comp}"
                st.session_state.edit_selected_candidates = {}
                st.rerun()
        with c3:
            if st.button("ยกเลิก", use_container_width=True, key=f"edit_cancel_candidates_{comp}"):
                st.session_state.pending_edit = None
                st.session_state.edit_selected_candidates = {}
                st.session_state.edit_targets = []
                st.rerun()
        return

    # Step 3B: One confirmation for all selected AI changes.
    if pending.startswith("confirm_"):
        preview = st.session_state.get("edit_preview", {})
        failed = st.session_state.get("edit_preview_failed", [])
        unchanged = st.session_state.get("edit_preview_unchanged", [])
        unselected = st.session_state.get("edit_preview_unselected", [])
        selected_candidates = st.session_state.get("edit_selected_candidates", {})
        st.markdown("**🔍 Preview การเปลี่ยนแปลงก่อนยืนยัน**")
        for idx in targets:
            name = selected_candidates.get(idx)
            nr = preview.get(idx)
            old_item = builds[idx]['build'][comp]
            old_name = old_item.get('name', '-')
            old_price = old_item.get('price', 0)
            if idx in unselected:
                st.info(f"**Build {idx+1}:** ยังไม่ได้เลือก Candidate — จะไม่เปลี่ยนแปลง Build นี้")
            elif nr:
                new_item = nr['build'][comp]
                new_price = new_item.get('price', 0)
                diff = new_price - old_price
                diff_text = f"เพิ่ม {diff:,.0f} บาท" if diff > 0 else (f"ลด {abs(diff):,.0f} บาท" if diff < 0 else "ราคาเท่าเดิม")
                st.success(f"**Build {idx+1}:** {old_name} → **{name}** | ราคา {old_price:,.0f} → {new_price:,.0f} บาท ({diff_text}) | ราคารวมใหม่ **{nr['total']:,.0f} บาท**")
            elif idx in unchanged:
                st.info(f"**Build {idx+1}:** ใช้ **{old_name}** อยู่แล้ว — ไม่มีการเปลี่ยนแปลง")
            else:
                st.error(f"Build {idx+1}: ไม่สามารถใช้ **{name}** โดยคงเงื่อนไขทั้งหมดได้")
        if failed:
            st.warning("Build ที่ไม่ผ่านจะยังไม่ถูกเปลี่ยน")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ ยืนยันการเปลี่ยนทั้งหมด", disabled=not preview, use_container_width=True, key=f"confirm_all_{comp}"):
                nb = list(builds)
                for idx, nr in preview.items():
                    nb[idx] = nr
                st.session_state.current_builds = nb
                changed = ", ".join(f"Build {i+1}" for i in preview)
                unchanged_text = ", ".join(f"Build {i+1}" for i in unchanged)
                failed_text = ", ".join(f"Build {i+1}" for i in failed)
                parts = [f"ยืนยันการเปลี่ยน {_component_label(comp)} สำเร็จใน **{changed}**"] if changed else []
                if unchanged_text:
                    parts.append(f"Build ที่ไม่มีการเปลี่ยนแปลง: **{unchanged_text}**")
                if failed_text:
                    parts.append(f"Build ที่ไม่ผ่าน: **{failed_text}**")
                st.session_state.pending_edit = None
                st.session_state.edit_selected_candidates = {}
                st.session_state.edit_preview = {}
                st.session_state.edit_preview_failed = []
                st.session_state.edit_preview_unchanged = []
                st.session_state.messages.append({"role":"assistant", "content":"\n".join(parts) + "\n\n" + make_reply(req)})
                st.rerun()
        with c2:
            if st.button("ยกเลิก", use_container_width=True, key=f"edit_cancel_confirm_{comp}"):
                st.session_state.pending_edit = f"candidates_{comp}"
                st.session_state.edit_preview = {}
                st.session_state.edit_preview_failed = []
                st.session_state.edit_preview_unchanged = []
                st.session_state.edit_preview_unselected = []
                st.rerun()
        return

    # Step 3C: Manual model entry. The actual text is handled by chat_input;
    # after it is resolved, a preview is shown here and one confirmation applies
    # the change to every selected Build that passes validation.
    if pending.startswith("manual_"):
        manual_preview = st.session_state.get("manual_preview", {})
        if manual_preview:
            st.markdown("**🔍 Preview การเปลี่ยนแปลงก่อนยืนยัน**")
            for idx in targets:
                name = manual_preview.get("name")
                nr = manual_preview.get(idx)
                old_item = builds[idx]['build'][comp]
                old_name = old_item.get('name', '-')
                old_price = old_item.get('price', 0)
                if nr:
                    new_item = nr['build'][comp]
                    new_price = new_item.get('price', 0)
                    diff = new_price - old_price
                    diff_text = f"เพิ่ม {diff:,.0f} บาท" if diff > 0 else (f"ลด {abs(diff):,.0f} บาท" if diff < 0 else "ราคาเท่าเดิม")
                    if old_name == name:
                        st.info(f"**Build {idx+1}:** ใช้ **{old_name}** อยู่แล้ว — ไม่มีการเปลี่ยนแปลง")
                    else:
                        st.success(f"**Build {idx+1}:** {old_name} → **{name}** | ราคา {old_price:,.0f} → {new_price:,.0f} บาท ({diff_text}) | ราคารวมใหม่ **{nr['total']:,.0f} บาท**")
                else:
                    st.error(f"Build {idx+1}: รุ่น **{name}** ไม่สามารถใช้ได้โดยคงเงื่อนไขทั้งหมด")
            c1, c2 = st.columns(2)
            with c1:
                valid = [i for i in targets if i in manual_preview and i != "name" and builds[i]['build'][comp].get('name') != manual_preview.get('name')]
                unchanged_manual = [i for i in targets if i in manual_preview and i != "name" and builds[i]['build'][comp].get('name') == manual_preview.get('name')]
                if st.button("✅ ยืนยันการเปลี่ยนทั้งหมด", disabled=not valid, use_container_width=True, key=f"confirm_manual_{comp}"):
                    nb = list(builds)
                    for idx in valid:
                        nb[idx] = manual_preview[idx]
                    st.session_state.current_builds = nb
                    changed = ", ".join(f"Build {i+1}" for i in valid)
                    unchanged_text = ", ".join(f"Build {i+1}" for i in unchanged_manual)
                    failed_manual = [i for i in targets if i not in manual_preview and i != "name"]
                    failed_text = ", ".join(f"Build {i+1}" for i in failed_manual)
                    parts = [f"ยืนยันการเปลี่ยน {_component_label(comp)} เป็น **{manual_preview['name']}** สำเร็จใน **{changed}**"] if changed else []
                    if unchanged_text: parts.append(f"Build ที่ไม่มีการเปลี่ยนแปลง: **{unchanged_text}**")
                    if failed_text: parts.append(f"Build ที่ไม่ผ่าน: **{failed_text}**")
                    st.session_state.pending_edit = None
                    st.session_state.manual_preview = {}
                    st.session_state.messages.append({"role":"assistant", "content":"\n".join(parts) + "\n\n" + make_reply(req)})
                    st.rerun()
            with c2:
                if st.button("ยกเลิก", use_container_width=True, key=f"cancel_manual_preview_{comp}"):
                    st.session_state.manual_preview = {}
                    st.session_state.pending_edit = f"manual_{comp}"
                    st.rerun()
        else:
            if comp == "ram":
                st.info("📝 **วิธีระบุ RAM:** พิมพ์ขนาดเป็น GB เช่น **32GB**, **64GB** หรือพิมพ์แค่ **64** ก็ได้ (ระบบจะตีความเป็น **64GB** ไม่ใช่ 64 บาท) หากต้องการระบุชนิด ให้พิมพ์เช่น **64GB DDR5**")
            elif comp == "storage":
                st.info("📝 **วิธีระบุ SSD/Storage:** พิมพ์ความจุ เช่น **512GB**, **1TB** หรือ **2TB**")
            else:
                st.info(f"📝 **วิธีระบุ {_component_label(comp)}:** พิมพ์ชื่อรุ่น เช่น ชื่อ CPU/GPU ที่ต้องการ ระบบจะตรวจทุก Build ที่เลือก แล้วให้กดยืนยันพร้อมกัน")
        if st.button("ย้อนกลับ", use_container_width=True, key=f"edit_back_manual_{comp}"):
            st.session_state.pending_edit = f"method_{comp}"
            st.session_state.manual_preview = {}
            st.rerun()


def show_usage_buttons(mode="choose"):
    """Render usage choices. In add mode, only show usages not already selected."""
    req = st.session_state.requirements
    selected = set(k for k in USAGE_KEYS if req.get(k))
    available = [(label, key, text) for label, key, text in USAGE_OPTIONS if key not in selected]
    if not available:
        return

    cols = st.columns(len(available))
    for col, (label, key, text) in zip(cols, available):
        if col.button(label, use_container_width=True, key=f"usage_{mode}_{key}"):
            req[key] = True
            st.session_state.requirements = req
            st.session_state.messages.append({"role": "user", "content": f"{text} (เลือกจากปุ่ม)"})
            reply = make_reply(req.copy())
            prefix = "รับ **{}** แล้วครับ".format(text)
            if len(usage_labels(req)) > 1:
                prefix += " และเพิ่มการใช้งานร่วมกันแล้วครับ"
            st.session_state.messages.append({"role": "assistant", "content": prefix + "\n\n" + reply})
            st.rerun()


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

req = st.session_state.requirements

# If the latest state has only a budget, ask fresh about usage. The old usage is cleared
# when the user enters a budget-only message (handled below).
if req.get("budget") and not usage_labels(req):
    st.info("มีงบประมาณแล้วครับ รอบนี้เป็น Requirement ใหม่ กรุณาเลือกการใช้งานก่อน ระบบจะไม่ใช้ลักษณะการใช้งานจากรอบก่อน")
    show_usage_buttons("choose")

# If a usage is known, offer adding another usage without requiring retyping.
req = st.session_state.requirements
if req.get("budget") and usage_labels(req):
    with st.expander("➕ เพิ่มการใช้งานอื่น", expanded=False):
        st.caption("เลือกเพิ่มได้ เช่น Gaming + เขียนโปรแกรม ระบบจะจัดอันดับชุดใหม่ตาม Requirement รวม")
        show_usage_buttons("add")

if usage_labels(req) and not req.get("budget"):
    st.info("ทราบลักษณะการใช้งานแล้วครับ ขอทราบงบประมาณก่อน เพื่อให้ AI จัดชุด PC ที่อยู่ในงบ")

show_component_edit_controls()

prompt = st.chat_input("เช่น งบ 30,000 เล่นเกมและเขียนโปรแกรม ขอ RTX 4060 RAM 32GB DDR5 SSD 1TB")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Natural-language budget questions are handled before component-edit
    # commands so phrases such as "จัดคอมงบต่ำสุดได้เท่าไหร่" do not become
    # accidental Requirement updates.
    budget_kind = budget_question_kind(prompt)
    if budget_kind:
        reply = budget_question_reply(budget_kind)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # Conversational component edits now use buttons for Build selection and
    # AI/manual choice. The chat input is only used for the manual model name.
    edit_component = None
    for c in ["cpu", "gpu", "ram", "storage"]:
        if _component_edit_command(prompt, c):
            edit_component = c
            break
    if edit_component:
        if not st.session_state.requirements.get("budget") or not usage_labels(st.session_state.requirements):
            reply = "ตอนนี้ยังไม่มีชุด PC ปัจจุบันให้ปรับครับ กรุณาระบุงบประมาณและการใช้งานก่อน"
        else:
            _current_builds(st.session_state.requirements)
            st.session_state.edit_component=edit_component
            st.session_state.edit_targets=[]
            st.session_state.edit_candidate=None
            st.session_state.pending_edit=f"select_build_{edit_component}"
            reply=f"ได้ครับ ต้องการเปลี่ยน **{_component_label(edit_component)}** ของ Build ไหนครับ?\n\nเลือกได้ทั้ง **Build 1 / Build 2 / Build 3 / ทุก Build** จากปุ่มด้านล่าง"
    elif st.session_state.pending_edit and st.session_state.pending_edit.startswith("manual_"):
        # Manual component changes are now previewed first. For multiple selected
        # Builds, the same entered model is tested against every target and the
        # user confirms all successful changes with one button.
        comp=st.session_state.pending_edit.split("_",1)[1]
        targets=st.session_state.get("edit_targets",list(range(len(_current_builds(st.session_state.requirements)))))
        if comp == "cpu":
            item=_find_cpu_from_text(prompt); name=item["name"] if item else None
        elif comp == "gpu":
            parsed=parse_requirement(prompt,{}) ; name=parsed.get("gpu")
        elif comp == "ram":
            rv=extract_ram_value(prompt); name=None
            # In manual RAM mode, a bare number means GB. For example,
            # "64" means 64GB; it must never be reinterpreted as a 64-baht
            # budget.
            if rv is None and re.fullmatch(r"\s*\d+\s*", str(prompt or "")):
                rv=int(str(prompt).strip())
            if rv:
                candidates=[x for x in PRODUCTS.get("ram",[]) if (x.get("capacity") or 0)==rv]
                typed = re.search(r"\bddr\s*(4|5)\b", str(prompt or "").lower())
                if typed:
                    wanted = "DDR" + typed.group(1)
                    typed_candidates = [x for x in candidates if str(x.get("ddr", "")).upper() == wanted]
                    if typed_candidates:
                        candidates = typed_candidates
                candidates = sorted(candidates, key=lambda x: x.get("price", 999999))
                name=candidates[0]["name"] if candidates else None
        else:
            m=re.search(r"(\d+(?:\.\d+)?)\s*(gb|tb)",str(prompt).lower())
            name=None
            if m:
                cap=int(float(m.group(1))*(1000 if m.group(2)=="tb" else 1))
                candidates=[x for x in PRODUCTS.get("storage",[]) if (x.get("capacity") or 0)>=cap and "SSD" in x.get("type","").upper()]
                if candidates: name=sorted(candidates,key=lambda x:x.get("price",999999))[0]["name"]
        if not name:
            reply=f"ยังหา {_component_label(comp)} ที่ระบุไม่พบครับ ลองพิมพ์ชื่อรุ่นหรือความจุที่มีอยู่ใน Dataset อีกครั้ง"
        else:
            builds=_current_builds(st.session_state.requirements)
            preview={"name":name}
            for idx in targets:
                if idx >= len(builds):
                    continue
                nr=_apply_component_to_build(st.session_state.requirements,builds[idx],comp,name)
                if nr:
                    preview[idx]=nr
            st.session_state.manual_preview=preview
            st.session_state.pending_edit=f"manual_{comp}"
            ok=[i for i in targets if i in preview]
            failed=[i+1 for i in targets if i not in preview]
            reply=f"พบรุ่น **{name}** แล้วครับ ระบบตรวจสอบ Build ที่เลือกเรียบร้อย\n\n"
            reply += f"ผ่าน: {', '.join(f'Build {i+1}' for i in ok) if ok else 'ไม่มี Build'}"
            if failed: reply += f"\nไม่ผ่าน: {', '.join(f'Build {i}' for i in failed)}"
            reply += "\n\nกรุณาตรวจสอบ Preview แล้วกด **ยืนยันการเปลี่ยนทั้งหมด** หากต้องการใช้การเปลี่ยนแปลง"
    elif st.session_state.pending_edit in {"ram_gb","storage_gb"} and not has_new_budget(prompt):
        # Legacy compatibility for existing conversations; new UI uses the
        # component editor buttons above.
        pending=st.session_state.pending_edit
        if pending=="ram_gb":
            rv=extract_ram_value(prompt)
            if rv:
                st.session_state.requirements["ram_gb"]=rv; _reset_current_builds(); st.session_state.pending_edit=None; reply=make_reply(st.session_state.requirements)
            else: reply="ขอขนาด RAM เป็น GB ก่อนครับ เช่น **32GB** หรือ **64GB**"
        else:
            m=re.search(r"(\d+(?:\.\d+)?)\s*(gb|tb)",str(prompt).lower())
            if m:
                val=int(float(m.group(1))*(1000 if m.group(2)=="tb" else 1)); st.session_state.requirements["storage_gb"]=val; st.session_state.requirements["storage_type"]="SSD"; _reset_current_builds(); st.session_state.pending_edit=None; reply=make_reply(st.session_state.requirements)
            else: reply="ขอความจุ Storage เป็น GB/TB ก่อนครับ เช่น **512GB**, **1TB** หรือ **2TB**"
    elif is_ram_edit_request(prompt) and not has_explicit_ram_value(prompt):
        if st.session_state.requirements.get("budget") and usage_labels(st.session_state.requirements):
            _current_builds(st.session_state.requirements); st.session_state.edit_component="ram"; st.session_state.edit_targets=[]; st.session_state.pending_edit="select_build_ram"
            reply="ได้ครับ เลือกก่อนว่าจะเปลี่ยน RAM ของ **Build 1 / Build 2 / Build 3 / ทุก Build** ครับ"
        else: reply="ตอนนี้ยังไม่มีชุด PC ปัจจุบันให้ปรับครับ กรุณาระบุงบประมาณและการใช้งานก่อน"
    elif _component_edit_command(prompt,"storage") and not re.search(r"\d+\s*(?:gb|tb)",prompt.lower()):
        if st.session_state.requirements.get("budget") and usage_labels(st.session_state.requirements):
            _current_builds(st.session_state.requirements); st.session_state.edit_component="storage"; st.session_state.edit_targets=[]; st.session_state.pending_edit="select_build_storage"
            reply="ได้ครับ เลือกก่อนว่าจะเปลี่ยน SSD/Storage ของ **Build 1 / Build 2 / Build 3 / ทุก Build** ครับ"
        else: reply="ตอนนี้ยังไม่มีชุด PC ปัจจุบันให้ปรับครับ กรุณาระบุงบประมาณและการใช้งานก่อน"
    # IMPORTANT: every message that introduces a budget starts a fresh request.
    # This fixes the bug where "งบ 40,000 เน้น CPU" inherited Gaming from the
    # previous request. The new message itself is the only source of the new
    # budget/usage/focus state.
    elif has_new_budget(prompt):
        parsed = parse_requirement(prompt, {})
        st.session_state.requirements = parsed
        st.session_state.pending_edit = None
        _reset_current_builds()
        req = st.session_state.requirements

        if usage_labels(req):
            reply = make_reply(req)
        else:
            focus_text = ""
            focus = []
            for key, label in [("focus_cpu", "CPU"), ("focus_gpu", "GPU"), ("focus_ram", "RAM"), ("focus_ssd", "SSD"), ("focus_hdd", "HDD")]:
                if req.get(key):
                    focus.append(label)
            if focus:
                focus_text = " และ **เน้น " + " + ".join(focus) + "**"
            reply = (
                f"รับงบประมาณ **{req.get('budget', 0):,} บาท**{focus_text} แล้วครับ\n\n"
                "รอบใหม่นี้จะไม่ใช้ลักษณะการใช้งานจากคำขอก่อนหน้า\n\n"
                "ก่อนจัดชุด ขอทราบว่าจะนำคอมไปใช้ทำอะไรครับ?\n\n"
                + make_reply(req)
            )
    else:
        # No new budget: this is an update to the current request.
        st.session_state.requirements = parse_requirement(prompt, st.session_state.requirements)
        _reset_current_builds()
        req = st.session_state.requirements

        if not req.get("budget") and not usage_labels(req):
            reply = ("ตอนนี้ยังไม่มีทั้งงบประมาณและลักษณะการใช้งานครับ\n\n"
                     "บอกผมได้เลย เช่น **งบ 30,000** หรือ **Gaming งบ 30,000**")
        elif req.get("budget") and not usage_labels(req):
            reply = (f"รับ Requirement เพิ่มแล้วครับ\n\n"
                     "ตอนนี้ยังไม่ได้ระบุการใช้งาน กรุณาเลือก Gaming / เขียนโปรแกรม / ใช้งานทั่วไป / ตัดต่อ / AI ก่อนครับ\n\n"
                     + make_reply(req))
        elif usage_labels(req) and not req.get("budget"):
            reply = "รับลักษณะการใช้งาน **" + " + ".join(usage_labels(req)) + "** แล้วครับ ขอทราบงบประมาณก่อน เช่น **20,000 / 30,000 / 40,000 บาท**"
        else:
            reply = make_reply(req)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

with st.sidebar:
    st.markdown("""<div class="kk-side-brand"><div class="brand">JIB · AI Recommendation</div><div class="line"></div></div>""", unsafe_allow_html=True)
    st.caption("Prototype · AI + ML + PC Compatibility")
    st.markdown("### 📌 Requirement ปัจจุบัน")
    req = st.session_state.requirements
    if req:
        budget = req.get("budget")
        if budget:
            st.metric("งบประมาณ", f"{budget:,.0f} บาท")
        usages = usage_labels(req)
        if usages:
            st.markdown("**การใช้งาน**")
            st.write(" + ".join(usages))
        focuses = []
        for key, label in [("focus_cpu", "CPU"), ("focus_gpu", "GPU"), ("focus_ram", "RAM"), ("focus_ssd", "SSD"), ("focus_hdd", "HDD")]:
            if req.get(key):
                focuses.append(label)
        if focuses:
            st.markdown("**เน้น**")
            st.write(" + ".join(focuses))
    else:
        st.info("ยังไม่มี Requirement\n\nเริ่มด้วยงบประมาณ เช่น 30,000 บาท")

    st.markdown("### 🧭 สิ่งที่ระบบวิเคราะห์")
    st.markdown("""<div class="kk-side-card"><div class="kk-side-label">AI ENGINE</div>🤖 AI Recommendation<br/>🧠 Machine Learning</div><div class="kk-side-card"><div class="kk-side-label">PC ANALYSIS</div>⚙️ CPU / GPU Balance<br/>🔌 PSU & Power Headroom<br/>❄️ CPU Cooling<br/>🧩 Mainboard / RAM Compatibility<br/>💾 Storage Strategy</div>""", unsafe_allow_html=True)

    if st.button("🔄 เริ่ม Requirement ใหม่", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.session_state.requirements = {}
        st.session_state.pending_edit = None
        st.session_state.current_builds = []
        st.rerun()

    st.caption("Prototype สำหรับการเรียนและการนำเสนอ\nJIB = กรณีศึกษา | กลุ่มผู้จัดทำ = ผู้พัฒนาและทดลอง")
