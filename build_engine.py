from pathlib import Path
import re
import math
import pickle
import itertools
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "build_ranker.pkl"

FILES = {
    "cpu": "JIB_Top50_CPU_with_TDP.csv",
    "gpu": "JIB_GPU_Top_50_with_TDP.csv",
    "ram": "RAM_Dataset_DDR4_DDR5_50_with_Product_URLs V.2_cleaned.csv",
    "mainboard": "JIB_Top50_Mainboards__cleaned.csv",
    "storage": "JIB_Top50_HDD_SSD_M2_2026-09-18_with_product_urls_cleaned.csv",
    "case": "50_case.csv",
    "psu": "PowerSupply_Direct_Links(PSU Direct Links)_cleaned.csv",
    "cooler": "CPU_Cooler_JIB_50_.csv",
    "monitor": "Monitor_Dataset_50_JIB_Verified_URLs_cleaned.csv",
}


def num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def price(v):
    return num(v)


def clean_text(v):
    return "" if v is None else str(v).strip()


def load_data():
    d = {}
    for key, filename in FILES.items():
        path = DATA_DIR / filename
        d[key] = pd.read_csv(path, encoding="utf-8-sig")
    return d


DATA = load_data()


def _integrated_gpu():
    """Virtual graphics entry for general-use builds when the selected CPU has integrated graphics."""
    return {
        "name": "Integrated Graphics (CPU) — ไม่ต้องซื้อการ์ดจอแยก",
        "chipset": "Integrated Graphics",
        "vram": "Shared Memory",
        "price": 0.0,
        "url": "",
        "tdp": 0.0,
        "integrated": True,
    }


def _cpu_has_integrated_graphics(cpu):
    n = clean_text(cpu.get("name")).upper().replace(" ", "")
    # Prototype heuristic based on common naming conventions in the dataset.
    if "RYZEN" in n and re.search(r"\dG(?:$|[^A-Z0-9])", n):
        return True
    if "RYZEN" in n and any(x in n for x in ["5600G", "5700G", "4600G", "4650G", "3400G", "3200G", "8500G", "8600G"]):
        return True
    if "INTEL" in n or n.startswith("CORE") or n.startswith("ULTRA"):
        # F/KF/KFT Intel desktop CPUs generally have no integrated graphics.
        # Search the CPU model portion rather than the end of the full CSV name.
        model_part = n.split("GHZ")[0]
        # Product names in this dataset place the clock immediately after the
        # model (e.g. 12100F3.3 GHz), so the suffix may be followed by a digit.
        # Treat Intel F/KF/KFT suffixes as "no iGPU" in that form as well.
        return not bool(re.search(r"\d(?:KF|KFT|F)(?:\d|[-(]|$)", model_part))
    return False


def _norm_socket(s):
    s = clean_text(s).upper().replace(" ", "")
    return s.replace("SOCKET", "")


def _norm_ddr(s):
    s = clean_text(s).upper()
    m = re.search(r"DDR[345]", s)
    return m.group(0) if m else ""


def _form_factors(s):
    s = clean_text(s).lower()
    vals = []
    for x in ["atx", "micro-atx", "mini-itx"]:
        if x in s:
            vals.append(x)
    return vals


def _cooler_sockets(s):
    raw = clean_text(s).upper().replace(" ", "")
    out = set(re.findall(r"LGA\d+|AM\d+|FM\d+|TR\d+", raw))
    for n in re.findall(r"(?<![A-Z])(?:1700|1200|1151|1150|1155|2066|2011-V3|2011)(?![A-Z])", raw):
        out.add("LGA" + n.replace("-V3", ""))
    return out


def _storage_gb(s):
    s = clean_text(s).upper().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TB|GB)", s)
    if not m:
        return None
    x = float(m.group(1))
    return x * 1000 if m.group(2) == "TB" else x


# Centralized usage vocabulary. The parser accepts Thai, English, and mixed input.
USAGE_ALIASES = {
    "gaming": [
        "เล่นเกม", "เล่นเกมส์", "เกมมิ่ง", "เกม", "gaming", "gaming pc", "game",
    ],
    "programming": [
        "เขียนโปรแกรม", "เขียนโค้ด", "โปรแกรมมิ่ง", "พัฒนาโปรแกรม", "พัฒนาซอฟต์แวร์",
        "โค้ด", "coding", "programming", "development", "software development",
        "web development", "developer", "dev",
    ],
    "editing": [
        "ตัดต่อวิดีโอ", "ตัดต่อ", "แต่งภาพ", "งานกราฟิก", "กราฟิก", "กราฟฟิก",
        "editing", "edit", "video editing", "photo editing", "graphics", "graphic",
        "rendering", "render",
    ],
    "general": [
        "ใช้งานทั่วไป", "งานทั่วไป", "ออฟฟิศ", "เอกสาร", "สำนักงาน",
        "general use", "general", "office", "work", "everyday use",
    ],
    "ai_ml": [
        "ปัญญาประดิษฐ์", "แมชชีนเลิร์นนิง", "เรียน ai", "งาน ai",
        "ai/ml", "ai ml", "artificial intelligence", "machine learning", "deep learning",
        "ai", "ml",
    ],
}


def _contains_any_usage(text, aliases):
    t = clean_text(text).lower()
    return any(alias in t for alias in aliases)


def detect_usages(text):
    """Return every usage found in one message; Thai/English/mixed are supported."""
    t = clean_text(text).lower()
    found = {}
    for key, aliases in USAGE_ALIASES.items():
        found[key] = any(alias in t for alias in aliases)

    # Avoid the generic AI alias "ai" matching unrelated ASCII words.
    if found.get("ai_ml"):
        ai_markers = [
            "ปัญญาประดิษฐ์", "แมชชีนเลิร์นนิง", "เรียน ai", "งาน ai", "ai/ml",
            "ai ml", "artificial intelligence", "machine learning", "deep learning"
        ]
        found["ai_ml"] = any(marker in t for marker in ai_markers) or bool(re.search(r"(?:^|\W)(?:ai|ml)(?:$|\W)", t))
    return {k: v for k, v in found.items() if v}


def parse_requirement(text, current=None):
    req = dict(current or {})
    t = clean_text(text).lower().replace(",", "")

    money = re.search(r"(?:งบ(?:ประมาณ)?\s*)?(\d+(?:\.\d+)?)\s*(บาท|฿|k\b)?(?!\d|\s*(?:gb|tb)\b)", t)
    if money:
        val = float(money.group(1))
        unit = money.group(2) or ""
        if unit == "k":
            val *= 1000
        req["budget"] = int(val)

    for usage_key in detect_usages(t):
        req[usage_key] = True

    gpu = re.search(r"(rtx\s*\d{3,4}(?:\s*(?:ti|super))?|rx\s*\d{3,4}(?:\s*(?:xt|xtx))?)", t)
    if gpu:
        req["gpu"] = re.sub(r"\s+", " ", gpu.group(1)).upper()

    ram = re.search(r"(?:ram|แรม)\s*(?:ขอ|เป็น|ให้เป็น|เอา)?\s*(\d+)\s*gb", t)
    if ram:
        req["ram_gb"] = int(ram.group(1))
    if "ddr5" in t:
        req["ram_type"] = "DDR5"
    elif "ddr4" in t:
        req["ram_type"] = "DDR4"

    storage = re.search(r"(?:ssd|m\.2|storage|พื้นที่)\s*(?:ขอ|เป็น|ให้เป็น|เอา)?\s*(\d+(?:\.\d+)?)\s*(gb|tb)", t)
    if storage:
        val = float(storage.group(1))
        if storage.group(2) == "tb":
            val *= 1000
        req["storage_gb"] = int(val)
        req["storage_type"] = "SSD" if "ssd" in storage.group(0) or "m.2" in storage.group(0) else ""

    if any(x in t for x in ["wifi", "wi-fi"]):
        req["wifi"] = True
    if "intel" in t:
        req["cpu_brand"] = "Intel"
    if "amd" in t or "ryzen" in t:
        req["cpu_brand"] = "AMD"

    focus_map = {
        "focus_cpu": ["เน้น cpu", "เน้นซีพียู", "เน้น cpu/ซีพียู", "focus cpu", "prioritize cpu"],
        "focus_gpu": ["เน้น gpu", "เน้นการ์ดจอ", "เน้นการ์ดแสดงผล", "focus gpu", "prioritize gpu"],
        "focus_ram": ["เน้น ram", "เน้นแรม", "focus ram", "prioritize ram"],
        "focus_ssd": ["เน้น ssd", "เน้น m.2", "เน้น nvme", "focus ssd", "prioritize ssd"],
        "focus_hdd": ["เน้น hdd", "เน้นฮาร์ดดิสก์", "เน้น hard disk", "focus hdd", "prioritize hdd"],
    }
    for key, words in focus_map.items():
        if any(w in t for w in words):
            req[key] = True

    return req


def _component_rows(req):
    d = DATA
    cpu = []
    for _, r in d["cpu"].iterrows():
        p = price(r.get("ราคา (บาท)"))
        if p is None:
            continue
        cpu.append({"name": clean_text(r.get("รุ่น / รายละเอียด")), "brand": clean_text(r.get("แบรนด์")),
                    "socket": _norm_socket(r.get("Socket")), "price": p, "url": clean_text(r.get("ลิงก์สินค้า JIB")),
                    "tdp": num(r.get("TDP (W)")) or 0,
                    "max_power": num(r.get("Max Power (W)")) or num(r.get("TDP (W)")) or 0,
                    "ram_native_max": num(r.get("RAM Native Max Speed (MHz)")) or 0})

    gpu = []
    for _, r in d["gpu"].iterrows():
        p = price(r.get("ราคา"))
        if p is None:
            continue
        gpu.append({"name": clean_text(r.get("ชื่อรุ่นสินค้า")), "chipset": clean_text(r.get("ชิปเซ็ต")),
                    "vram": clean_text(r.get("หน่วยความจำ (VRAM)")), "price": p, "url": clean_text(r.get("ลิงก์สินค้า")),
                    "tdp": num(r.get("TDP (W)")) or 0})

    ram = []
    for _, r in d["ram"].iterrows():
        p = price(r.get("Price_THB"))
        if p is None:
            continue
        ram.append({"name": f"{clean_text(r.get('Brand'))} {clean_text(r.get('Model'))}", "brand": clean_text(r.get("Brand")),
                    "ddr": _norm_ddr(r.get("DDR")), "capacity": num(r.get("Capacity_GB")),
                    "bus": num(r.get("Bus_MHz")), "xmp": clean_text(r.get("XMP")).upper(),
                    "expo": clean_text(r.get("EXPO")).upper(), "price": p, "url": clean_text(r.get("Product_URL"))})

    mb = []
    for _, r in d["mainboard"].iterrows():
        p = price(r.get("ราคาโดยประมาณ (บาท)"))
        if p is None:
            continue
        mb.append({"name": clean_text(r.get("รุ่นเมนบอร์ด (Model)")), "brand": clean_text(r.get("ยี่ห้อ (Brand)")),
                   "socket": _norm_socket(r.get("ซ็อกเก็ต (Socket)")), "ddr": _norm_ddr(r.get("ประเภทแรม (RAM)")),
                   "form": clean_text(r.get("ขนาด (Form Factor)")), "wifi": clean_text(r.get("Wi-Fi")),
                   "ram_max_speed": num(r.get("RAM Max Speed (MHz)")) or 0,
                   "ram_max_capacity": num(r.get("RAM Max Capacity (GB)")) or 0,
                   "ram_profile": clean_text(r.get("RAM Profile Support")).upper(),
                   "price": p, "url": clean_text(r.get("URL ลิงก์สินค้า (JIB)"))})

    storage = []
    for _, r in d["storage"].iterrows():
        p = price(r.get("ราคา (บาท)"))
        cap = _storage_gb(r.get("ความจุ"))
        if p is None or cap is None:
            continue
        storage.append({"name": clean_text(r.get("รุ่นสินค้า")), "type": clean_text(r.get("ประเภท")),
                        "capacity": cap, "price": p, "url": clean_text(r.get("Source URL"))})

    cases = []
    for _, r in d["case"].iterrows():
        p = price(r.get("ราคา (บาท)"))
        if p is None:
            continue
        cases.append({"name": clean_text(r.get("รุ่นเคส")), "form": clean_text(r.get("ขนาดเมนบอร์ดที่รองรับ")),
                      "price": p, "url": clean_text(r.get("ลิงก์หน้าสินค้า / หน้าค้นหา JIB"))})

    psus = []
    for _, r in d["psu"].iterrows():
        p = price(r.get("Estimated Price (THB)"))
        w = num(r.get("Wattage"))
        if p is None or w is None:
            continue
        psus.append({"name": clean_text(r.get("Product Name")), "watt": w, "efficiency": clean_text(r.get("Efficiency")),
                     "price": p, "url": clean_text(r.get("Product Link"))})

    coolers = []
    for _, r in d["cooler"].iterrows():
        p = price(r.get("ราคา (บาท)"))
        if p is None:
            continue
        coolers.append({"name": f"{clean_text(r.get('ยี่ห้อ'))} {clean_text(r.get('รุ่น'))}",
                        "socket": clean_text(r.get("Socket")), "sockets": _cooler_sockets(r.get("Socket")),
                        "type": clean_text(r.get("ประเภท")),
                        "price": p, "url": clean_text(r.get("ลิงก์ JIB"))})

    monitors = []
    for _, r in d["monitor"].iterrows():
        p = price(r.get("Price_THB"))
        if p is None:
            continue
        monitors.append({"name": f"{clean_text(r.get('Brand'))} {clean_text(r.get('Model'))}",
                         "size": num(r.get("Size_Inch")), "refresh": num(r.get("Refresh_Rate_Hz")),
                         "resolution": clean_text(r.get("Resolution")), "price": p, "url": clean_text(r.get("Product_URL"))})

    return {"cpu": cpu, "gpu": gpu, "ram": ram, "mainboard": mb, "storage": storage,
            "case": cases, "psu": psus, "cooler": coolers, "monitor": monitors}


PRODUCTS = _component_rows({})


def _gpu_power_estimate(gpu):
    """GPU board power from the dataset TDP/TBP column."""
    value = num(gpu.get("tdp"))
    if value is not None and value > 0:
        return value
    return 130


def _cpu_power_estimate(cpu):
    """CPU sustained/base power proxy from dataset TDP."""
    value = num(cpu.get("tdp"))
    if value is not None and value > 0:
        return value
    return 65


def _cpu_max_power(cpu):
    """CPU maximum turbo/PPT proxy used for PSU and cooling decisions."""
    value = num(cpu.get("max_power"))
    if value is not None and value > 0:
        return value
    return _cpu_power_estimate(cpu)


def _cpu_cooling_requirement(cpu):
    """Cooling class derived from CPU TDP and maximum power.
    Very high-power CPUs require closed-loop liquid cooling in this prototype;
    lower tiers can use appropriately rated air coolers.
    """
    tdp = _cpu_power_estimate(cpu)
    max_power = _cpu_max_power(cpu)

    if max_power >= 220 or tdp >= 180:
        return {"min_capacity": 240, "liquid_only": True,
                "label": "ต้องใช้ชุดน้ำปิด 240mm ขึ้นไป"}
    if max_power >= 180 or tdp >= 120:
        return {"min_capacity": 190, "liquid_only": False,
                "label": "ควรใช้ Air Cooler ระดับสูง หรือชุดน้ำปิด 240mm ขึ้นไป"}
    if max_power >= 140 or tdp >= 105:
        return {"min_capacity": 150, "liquid_only": False,
                "label": "ควรใช้ Air Cooler ระดับกลาง-สูง หรือชุดน้ำปิด"}
    if max_power >= 90 or tdp >= 65:
        return {"min_capacity": 100, "liquid_only": False,
                "label": "ใช้ Air Cooler ที่เหมาะสมได้"}
    return {"min_capacity": 80, "liquid_only": False,
            "label": "Air Cooler ระดับพื้นฐานเพียงพอ"}


def _cooler_capacity(cooler):
    typ = clean_text(cooler.get("type")).lower()
    name = clean_text(cooler.get("name")).upper()
    if "liquid" in typ:
        if "360" in name:
            return 260
        if "280" in name:
            return 250
        if "240" in name:
            return 220
        return 210
    if any(x in name for x in [
        "NH-D15", "ASSASSIN IV", "NH-U12A", "DARK ROCK 5",
        "AK620", "FORZA 85", "BOREAS P2-62D"
    ]):
        return 200
    if any(x in name for x in [
        "NH-U12S", "NH-L9X65", "AS500", "FORZA 50", "FREEZER 36",
        "T120", "AK400", "A410", "BOREAS E2-410", "ASTRIA 200",
        "CNPS9X"
    ]):
        return 150
    return 95


def _cooler_is_suitable(cpu, cooler):
    req = _cpu_cooling_requirement(cpu); capacity = _cooler_capacity(cooler)
    is_liquid = "liquid" in clean_text(cooler.get("type")).lower()
    if req["liquid_only"] and not is_liquid: return False
    return capacity >= req["min_capacity"]

def _estimated_raw_power_watt(build):
    """Approximate sustained system load before PSU headroom."""
    cpu_max = _cpu_max_power(build["cpu"])
    gpu_tdp = _gpu_power_estimate(build["gpu"])
    ram_w = 20 if (build["ram"].get("capacity") or 0) > 32 else 10
    storage_w = 15
    motherboard_misc = 70
    return cpu_max + gpu_tdp + ram_w + storage_w + motherboard_misc


def _required_psu_watt(build):
    """Estimate a practical PSU target with useful reserve without oversizing.

    The estimate itself includes headroom, so the selector chooses the closest
    real PSU at or above this target instead of automatically jumping another
    large wattage tier.
    """
    cpu_max = _cpu_max_power(build["cpu"])
    gpu_tdp = _gpu_power_estimate(build["gpu"])
    ram_w = 20 if (build["ram"].get("capacity") or 0) > 32 else 10
    storage_w = 15
    motherboard_misc = 70
    raw = cpu_max + gpu_tdp + ram_w + storage_w + motherboard_misc
    calculated = math.ceil((raw * 1.30) / 50) * 50

    # Practical GPU floor. This is a floor, not a request to buy an oversized PSU.
    if gpu_tdp <= 130:
        gpu_min = 550
    elif gpu_tdp <= 180:
        gpu_min = 550
    elif gpu_tdp <= 250:
        gpu_min = 650
    elif gpu_tdp <= 300:
        gpu_min = 750
    elif gpu_tdp <= 360:
        gpu_min = 850
    else:
        gpu_min = 1000

    # CPU power is already included in the calculated system load above.
    # Do not impose a blanket 750/850W floor just because the CPU is high-end;
    # that would waste budget on an iGPU-only or low-GPU build.
    target = max(calculated, gpu_min)
    return int(math.ceil(target / 50.0) * 50)

def _select_psu_for_build(build, budget=None):
    """Choose a cost-aware PSU close to the required target.

    The PSU must have reserve, but the recommender should not consume budget on
    an unnecessarily large wattage tier. Prefer the smallest real PSU that is
    at least 50W above the calculated target; if that is unavailable, use the
    smallest PSU at/above the target. Among equal wattages prefer lower price.
    """
    required = _required_psu_watt(build)
    candidates = [x for x in PRODUCTS["psu"] if x.get("watt", 0) >= required]
    if not candidates:
        return None
    gpu_tdp = _gpu_power_estimate(build.get("gpu", {}))
    # Mainstream GPUs do not need an extra PSU tier just for the sake of a
    # fixed reserve. Higher-power GPUs get an additional 50W tier where the
    # dataset supports it. This creates proportionate 550/650/750/850W choices.
    extra_reserve = 50 if (gpu_tdp > 220 and required < 750) else 0
    with_reserve = [x for x in candidates if x.get("watt", 0) >= required + extra_reserve]
    pool = with_reserve or candidates
    # Prefer the closest wattage tier first. Price breaks ties so PSU cost does
    # not unnecessarily take money away from CPU/GPU/RAM/Storage.
    pool.sort(key=lambda x: (x.get("watt", 0), x.get("price", float("inf"))))
    return pool[0]

def _budget_utilization_score(total, budget):
    if not budget or budget <= 0:
        return 0.5
    ratio = total / budget
    if ratio > 1.0:
        return 0.0
    # A large unused budget is only acceptable when the build is already strong.
    # This rewards useful budget use without forcing the system to spend every baht.
    if ratio >= 0.90:
        return 1.0
    if ratio >= 0.82:
        return 0.92
    if ratio >= 0.75:
        return 0.80
    if ratio >= 0.68:
        return 0.62
    if ratio >= 0.60:
        return 0.45
    return 0.30

def _minimum_cpu_performance_for_usage(req, gpus):
    """Minimum CPU tier for GPU-heavy workloads to avoid obvious imbalance."""
    if not (req.get("gaming") or req.get("editing") or req.get("ai_ml") or req.get("focus_gpu")):
        if req.get("focus_cpu"):
            budget = req.get("budget") or 0
            return 80.0 if budget >= 35000 else (65.0 if budget >= 25000 else 55.0)
        return 0.0

    gpu_scores = [_gpu_performance_score(g) for g in gpus if not g.get("integrated")]
    max_gpu = max(gpu_scores, default=0.0)
    if max_gpu >= 85:
        floor = 72.0
    elif max_gpu >= 75:
        floor = 68.0
    elif max_gpu >= 65:
        floor = 62.0
    elif max_gpu >= 55:
        floor = 58.0
    else:
        floor = 52.0

    if req.get("focus_cpu"):
        floor = max(floor, 80.0 if (req.get("budget") or 0) >= 35000 else 65.0)
    return floor


def _usage_budget_policy(req):
    """Return practical budget-share caps for core components.

    These are prototype allocation guardrails, not benchmark claims. They keep a
    single expensive part from consuming the budget when it is not the stated focus.
    """
    active = [k for k in ["gaming", "programming", "editing", "general", "ai_ml"] if req.get(k)]
    budget = req.get("budget") or 0

    if req.get("focus_cpu"):
        cpu_cap = 0.42
    elif "general" in active and len(active) == 1:
        cpu_cap = 0.33
    elif "gaming" in active:
        cpu_cap = 0.38
    elif "editing" in active:
        cpu_cap = 0.38
    elif "ai_ml" in active:
        cpu_cap = 0.42
    elif "programming" in active:
        cpu_cap = 0.40
    else:
        cpu_cap = 0.40

    if req.get("focus_cpu"):
        mb_cap = 0.20
    elif "general" in active and len(active) == 1:
        mb_cap = 0.18
    elif "gaming" in active:
        mb_cap = 0.20
    elif "programming" in active:
        mb_cap = 0.20
    elif "editing" in active:
        mb_cap = 0.22
    elif "ai_ml" in active:
        mb_cap = 0.22
    else:
        mb_cap = 0.20

    # Explicit RAM requests get more room, but still prefer value over luxury kits.
    # When multiple workloads are requested, programming/editing/AI workloads
    # should not inherit the lower Gaming-only RAM cap. A 32GB kit must remain
    # reachable in a mixed-workload build when it is reasonably priced.
    if req.get("focus_ram"):
        ram_cap = 0.40
    elif req.get("ram_gb"):
        ram_cap = 0.28
    elif "general" in active and len(active) == 1:
        ram_cap = 0.15
    elif len(active) > 1 and any(k in active for k in ["programming", "editing", "ai_ml"]):
        ram_cap = 0.25
    elif "gaming" in active:
        ram_cap = 0.18
    elif "programming" in active or "editing" in active or "ai_ml" in active:
        ram_cap = 0.25
    else:
        ram_cap = 0.20

    return {"budget": budget, "cpu_cap": cpu_cap, "mb_cap": mb_cap, "ram_cap": ram_cap}


def _budget_share(price_value, budget):
    if not budget or budget <= 0:
        return 0.0
    return max(0.0, float(price_value or 0)) / float(budget)


def _budget_component_penalty(build, req):
    """Penalty for disproportionate spending on non-focused components."""
    policy = _usage_budget_policy(req)
    budget = policy["budget"]
    if not budget:
        return 0.0

    penalty = 0.0
    cpu_share = _budget_share(build["cpu"].get("price"), budget)
    mb_share = _budget_share(build["mainboard"].get("price"), budget)
    ram_share = _budget_share(build["ram"].get("price"), budget)
    psu_share = _budget_share(build["psu"].get("price"), budget)
    storage_share = _budget_share(build["storage"].get("price"), budget)

    if not req.get("focus_cpu") and cpu_share > policy["cpu_cap"]:
        penalty += min(7.0, (cpu_share - policy["cpu_cap"]) * 45.0)
    if mb_share > policy["mb_cap"]:
        penalty += min(6.0, (mb_share - policy["mb_cap"]) * 35.0)
    if not req.get("focus_ram") and ram_share > policy["ram_cap"]:
        penalty += min(6.0, (ram_share - policy["ram_cap"]) * 30.0)
    if not req.get("focus_ssd") and storage_share > 0.15:
        penalty += min(4.0, (storage_share - 0.15) * 25.0)

    required = _required_psu_watt(build)
    reserve = build["psu"].get("watt", 0) - required
    if reserve > 150 and psu_share > 0.08:
        penalty += min(4.0, (reserve - 150) / 40.0)
    return penalty


def _component_price_ok(component, cap_share, budget):
    if not budget:
        return True
    return (component.get("price", 0) or 0) <= budget * cap_share

def _build_diversity_signature(build):
    return (
        round(_cpu_performance_score(build["cpu"]) / 5) * 5,
        round(_gpu_performance_score(build["gpu"]) / 5) * 5,
        build["ram"].get("capacity"),
        build["ram"].get("ddr"),
        build["storage"].get("capacity"),
        build["storage"].get("type"),
    )

def _cpu_performance_score(cpu):
    """Relative CPU tier for prototype balancing; not a benchmark result."""
    n=clean_text(cpu.get("name")).upper().replace(" ","")
    anchors={"RYZEN33200G":42,"RYZEN53400G":48,"RYZEN54500":52,"RYZEN55500":60,"RYZEN55600":68,"RYZEN57500F":72,"RYZEN57600":76,"RYZEN57600X":80,"RYZEN58600G":72,"RYZEN58500G":68,"RYZEN59600X":82,"RYZEN77800X3D":92,"RYZEN79800X3D":96,"RYZEN79700X":88,"RYZEN79850X3D":98,"RYZEN99950X":98,"RYZEN99950X3D":100,"I312100":48,"I312100F":48,"I314100":58,"I314100F":58,"I512400":66,"I512400F":66,"I514400":76,"I514400F":76,"I514500":80,"I514600KF":86,"I712700KF":84,"I714700":92,"I714700F":92,"I714700KF":94,"I914900":97,"I914900K":99,"I914900KF":99,"COREULTRA5245K":86,"COREULTRA5245KF":86,"COREULTRA7225":70,"COREULTRA7265":90,"COREULTRA7265K":94,"COREULTRA7265KF":94,"COREULTRA7270KPLUS":96,"COREULTRA9285K":100}
    for token,value in anchors.items():
        if token in n:return float(value)
    if "I9" in n or "RYZEN9" in n or "COREULTRA9" in n:return 95.0
    if "I7" in n or "RYZEN7" in n or "COREULTRA7" in n:return 82.0
    if "I5" in n or "RYZEN5" in n or "COREULTRA5" in n:return 68.0
    if "I3" in n or "RYZEN3" in n:return 48.0
    return 30.0

def _gpu_performance_score(gpu):
    token = clean_text(gpu.get("chipset") or gpu.get("name")).upper().replace(" ", "")
    m = re.search(r"(RTX|RX)(\d{3,4})", token)
    if not m:
        return _gpu_power_estimate(gpu) / 4.0
    fam, num_ = m.group(1), int(m.group(2))
    # Coarse tier score used only for balance/ranking.
    if fam == "RTX":
        table = {3050: 35, 4060: 55, 4060: 55, 4070: 72, 4080: 90, 4090: 100,
                 5060: 62, 5070: 78, 5080: 94, 5090: 100}
    else:
        table = {6600: 42, 7600: 55, 7700: 68, 7800: 78, 7900: 92}
    base = table.get(num_, min(100, 30 + (num_ % 1000) / 10))
    if any(x in token for x in ["TI", "SUPER", "XT", "XTX"]): base += 4
    return min(base, 100.0)

def _balance_score(build):
    """Penalize obvious CPU/GPU imbalance for gaming-style builds."""
    cpu = _cpu_performance_score(build["cpu"])
    gpu = _gpu_performance_score(build["gpu"])
    gap = gpu - cpu
    if gap <= 8:
        return 1.0
    if gap <= 18:
        return 0.90
    if gap <= 28:
        return 0.72
    if gap <= 40:
        return 0.45
    return 0.20

def _balance_note(build):
    cpu = _cpu_performance_score(build["cpu"])
    gpu = _gpu_performance_score(build["gpu"])
    gap = gpu - cpu
    if gap > 28:
        return "CPU ต่ำกว่าระดับ GPU มาก มีความเสี่ยงเกิดคอขวดฝั่ง CPU ในเกมที่ใช้ CPU สูง"
    if gap > 18:
        return "CPU ต่ำกว่า GPU ค่อนข้างมาก ควรพิจารณาอัป CPU เพื่อให้ชุดสมดุลขึ้น"
    return "CPU/GPU อยู่ในระดับสมดุลสำหรับการจัดชุดแบบ Prototype"




def _ram_profile_match(ram, mb):
    """Check whether the motherboard profile support matches the RAM's rated profile."""
    profile = clean_text(mb.get("ram_profile")).upper()
    if not profile:
        return True, ""
    has_xmp = bool(clean_text(ram.get("xmp")).upper() and clean_text(ram.get("xmp")).upper() not in {"NAN", "NONE"})
    has_expo = bool(clean_text(ram.get("expo")).upper() and clean_text(ram.get("expo")).upper() not in {"NAN", "NONE"})
    if has_xmp and "XMP" in profile:
        return True, "XMP"
    if has_expo and "EXPO" in profile:
        return True, "EXPO"
    if not has_xmp and not has_expo:
        return True, "JEDEC"
    return False, "RAM profile ไม่ตรงกับที่ Mainboard รองรับ"


def _effective_cpu_native_ram_speed(cpu, mb):
    """Return a platform-aware native RAM ceiling for the prototype.

    The CSV stores one CPU-level native speed, but a CPU platform can support
    different JEDEC ceilings depending on DDR4 vs DDR5. This helper keeps the
    existing CSV field while avoiding a misleading "native" label for DDR4/DDR5
    combinations where the generic CSV value is higher than the board's RAM type.
    """
    native = num(cpu.get("ram_native_max")) or 0
    ddr = clean_text(mb.get("ddr")).upper()
    n = clean_text(cpu.get("name")).upper().replace(" ", "")
    if "DDR4" in ddr:
        if "INTEL" in n and re.search(r"I[3579]-?(12|13|14)\d{3}", n):
            return min(native, 3200) if native else 3200
        if "RYZEN" in n and re.search(r"(3200G|3400G|[34579]\d{3})", n):
            return min(native, 3200) if native else 3200
    elif "DDR5" in ddr:
        if "INTEL" in n:
            if re.search(r"I[3579]-?12\d{3}", n):
                return min(native, 4800) if native else 4800
            if re.search(r"I[3579]-?(13|14)\d{3}", n):
                return min(native, 5600) if native else 5600
        if "RYZEN" in n:
            if re.search(r"RYZEN[0-9]*[ ]?[3579][ ]?(7|8)\d{3}", n):
                return min(native, 5200) if native else 5200
            if re.search(r"RYZEN[0-9]*[ ]?[3579][ ]?9\d{3}", n):
                return min(native, 5600) if native else 5600
    return native


def _ram_mainboard_fit(cpu, mb, ram):
    """Validate RAM type, board ceiling, CPU native ceiling and memory profile.
    RAM above the CPU native speed is allowed when the motherboard supports the
    speed and the kit has a usable XMP/EXPO profile; it is labeled as OC rather
    than treated as a guaranteed native speed.
    """
    reasons = []
    if mb.get("ddr") and ram.get("ddr") and mb["ddr"] != ram["ddr"]:
        return False, ["ชนิด RAM ไม่ตรงกับ Mainboard"], 0
    speed = num(ram.get("bus")) or 0
    max_speed = num(mb.get("ram_max_speed")) or 0
    if speed and max_speed and speed > max_speed:
        return False, [f"RAM {speed:.0f}MHz สูงกว่าเพดาน Mainboard {max_speed:.0f}MHz"], max_speed - speed
    capacity = num(ram.get("capacity")) or 0
    max_capacity = num(mb.get("ram_max_capacity")) or 0
    if capacity and max_capacity and capacity > max_capacity:
        return False, [f"RAM {capacity:.0f}GB สูงกว่าความจุสูงสุดของ Mainboard {max_capacity:.0f}GB"], max_speed - speed

    profile_ok, profile = _ram_profile_match(ram, mb)
    cpu_native = _effective_cpu_native_ram_speed(cpu, mb)
    if speed and cpu_native and speed > cpu_native:
        if not profile_ok:
            reasons.append("RAM ความเร็วสูงกว่าค่า Native ของ CPU แต่ไม่มี XMP/EXPO ที่ Mainboard รองรับ")
        else:
            reasons.append(f"RAM {speed:.0f}MHz ทำงานเหนือ CPU Native {cpu_native:.0f}MHz ผ่าน {profile}")
    else:
        reasons.append(f"RAM {speed:.0f}MHz อยู่ในช่วง Native ของแพลตฟอร์ม CPU/Mainboard")
    if not profile_ok:
        reasons.append(profile)
    elif profile:
        reasons.append(f"Profile: {profile}")
    if speed and max_speed:
        reasons.append(f"Mainboard สูงสุด {max_speed:.0f}MHz")
    return True, reasons, max_speed - speed if max_speed else 0


def _mainboard_quality_score(mb):
    """Relative motherboard/platform tier for the prototype.
    This is a suitability tier, not a benchmark. It prevents a high-power CPU
    from being paired with an entry-level board merely because socket/DDR match.
    """
    n = clean_text(mb.get("name")).upper().replace(" ", "")
    desc = clean_text(mb.get("description", "")).lower()
    score = 35.0

    # Chipset/platform baseline.
    if "H610" in n:
        score = 20.0
    elif "H510" in n:
        score = 18.0
    elif "A520" in n:
        score = 28.0
    elif "B450" in n:
        score = 34.0
    elif "B550" in n:
        score = 45.0
    elif "B650" in n:
        score = 50.0
    elif "B850" in n:
        score = 55.0
    elif "B760" in n:
        score = 48.0
    elif "B860" in n:
        score = 50.0
    elif "Z690" in n or "Z790" in n:
        score = 68.0
    elif "Z890" in n:
        score = 70.0
    elif "X670" in n:
        score = 68.0
    elif "X870E" in n:
        score = 82.0
    elif "X870" in n:
        score = 70.0

    # Board series / VRM / feature clues from the dataset.
    if any(x in n for x in ["STEELLEGEND", "GAMINGPLUS", "TUF", "MORTAR", "AORUS", "STRIX", "EDGE", "GODLIKE", "NOVA"]):
        score += 10
    elif "PRO RS" in clean_text(mb.get("name")).upper():
        score += 7
    elif "PRO B760M-A" in n or "PRO B850M-A" in n:
        score += 7
    if any(x in desc for x in ["vrm แข็งแกร่ง", "ภาคจ่ายไฟดี", "ภาคจ่ายไฟแน่น", "ระบายความร้อนดี", "เล่นเกมหนัก", "ระดับพรีเมียม"]):
        score += 7
    if "WIFI" in n:
        score += 2
    if "ATX" in clean_text(mb.get("form")).upper():
        score += 2

    # Very low-cost entry boards should not masquerade as high-tier boards.
    p = mb.get("price", 0) or 0
    if p < 2000:
        score -= 3
    elif p >= 6000:
        score += 3
    return max(0.0, min(100.0, score))


def _mainboard_requirement(cpu, gpu=None, ram=None):
    """Minimum platform tier based primarily on CPU power, then overall build."""
    max_power = _cpu_max_power(cpu)
    tdp = _cpu_power_estimate(cpu)
    gpu_score = _gpu_performance_score(gpu) if gpu else 0

    # CPU power is the main driver. The thresholds intentionally keep basic
    # boards available for mainstream CPUs while filtering entry-level boards
    # from high-power CPU builds.
    if max_power >= 220 or tdp >= 180:
        minimum = 75
        label = "เมนบอร์ดระดับสูงสำหรับ CPU กำลังสูง"
    elif max_power >= 180 or tdp >= 120:
        minimum = 55
        label = "เมนบอร์ดระดับกลางขึ้นไปสำหรับ CPU กำลังสูง"
    elif max_power >= 140 or tdp >= 105:
        minimum = 45
        label = "เมนบอร์ดระดับกลางที่เหมาะกับ CPU"
    else:
        minimum = 25
        label = "เมนบอร์ดระดับพื้นฐาน/กลางเหมาะสม"

    # A very strong GPU does not force an expensive motherboard, but it raises
    # the platform floor slightly when paired with a strong CPU.
    if gpu_score >= 85 and max_power >= 140:
        minimum += 5
    return minimum, label


def _mainboard_is_suitable(build):
    mb = build["mainboard"]
    minimum, label = _mainboard_requirement(build["cpu"], build.get("gpu"), build.get("ram"))
    score = _mainboard_quality_score(mb)
    if score < minimum:
        return False, f"Mainboard ระดับไม่เหมาะกับชุดนี้: ได้ {score:.0f}/100 แต่ต้องการอย่างน้อย {minimum:.0f}/100 ({label})"
    return True, ""


def _mainboard_fit_score(build):
    minimum, _ = _mainboard_requirement(build["cpu"], build.get("gpu"), build.get("ram"))
    score = _mainboard_quality_score(build["mainboard"])
    if score >= minimum:
        # Small reward for useful headroom, but don't reward luxury boards heavily.
        return min(1.0, 0.75 + (score - minimum) / 100.0)
    return max(0.0, score / max(minimum, 1.0))

def compatibility(build):
    reasons = []
    ok = True
    cpu, mb, ram, case, psu, cooler = build["cpu"], build["mainboard"], build["ram"], build["case"], build["psu"], build["cooler"]

    if cpu["socket"] and mb["socket"] and cpu["socket"] != mb["socket"]:
        ok = False; reasons.append("CPU กับ Mainboard ใช้ Socket ไม่ตรงกัน")
    if mb["ddr"] and ram["ddr"] and mb["ddr"] != ram["ddr"]:
        ok = False; reasons.append("ชนิด RAM ไม่ตรงกับ Mainboard")
    mb_ok, mb_reason = _mainboard_is_suitable(build)
    if not mb_ok:
        ok = False; reasons.append(mb_reason)
    ram_ok, ram_reasons, _ = _ram_mainboard_fit(cpu, mb, ram)
    if not ram_ok:
        ok = False; reasons.extend(ram_reasons)
    elif ram_reasons:
        # Profile mismatch is not automatically fatal because JEDEC fallback can still boot;
        # keep it visible for ranking/reasons instead of rejecting a usable kit outright.
        if any("ไม่ตรง" in x for x in ram_reasons):
            reasons.extend(ram_reasons)
    forms = [x.lower() for x in _form_factors(case["form"])]
    mb_form = mb["form"].lower()
    if mb_form and forms and not any(x in mb_form for x in forms):
        if not ("micro-atx" in mb_form and "micro-atx" in forms):
            ok = False; reasons.append("ขนาด Mainboard ไม่ตรงกับ Case ที่ระบุไว้")
    if cooler["sockets"] and cpu["socket"] and cpu["socket"] not in cooler["sockets"]:
        ok = False; reasons.append("CPU Cooler ไม่รองรับ Socket ของ CPU")
    if not _cooler_is_suitable(cpu, cooler):
        ok = False; reasons.append(f"CPU Cooler ไม่เหมาะกับภาระความร้อนของ CPU: {_cpu_cooling_requirement(cpu)["label"]}")

    required = _required_psu_watt(build)
    if psu["watt"] < required:
        ok = False; reasons.append(f"PSU {psu['watt']:.0f}W ต่ำกว่าค่าขั้นต่ำที่ระบบประเมิน {required:.0f}W")

    return ok, reasons, required

def _usage_score(build, req):
    """Transparent usage-fit score. Returns 0..1."""
    scores = []
    if req.get("gaming"):
        gpu = _gpu_performance_score(build["gpu"])
        cpu = _cpu_performance_score(build["cpu"])
        ram = build["ram"].get("capacity") or 0
        scores.append(
            min(1.0, gpu / 100.0) * 0.42 +
            min(1.0, cpu / 100.0) * 0.28 +
            min(1.0, ram / 32.0) * 0.10 +
            _balance_score(build) * 0.20
        )
    if req.get("programming"):
        ram = build["ram"].get("capacity") or 0
        cpu = _cpu_power_estimate(build["cpu"])
        storage = build["storage"].get("capacity") or 0
        scores.append(min(1.0, ram / 32) * .40 + min(1.0, cpu / 140) * .35 + min(1.0, storage / 1000) * .25)
    if req.get("editing"):
        ram = build["ram"].get("capacity") or 0
        gpu = _gpu_power_estimate(build["gpu"])
        storage = build["storage"].get("capacity") or 0
        scores.append(min(1.0, ram / 32) * .35 + min(1.0, gpu / 300) * .40 + min(1.0, storage / 2000) * .25)
    if req.get("ai_ml"):
        ram = build["ram"].get("capacity") or 0
        gpu = _gpu_power_estimate(build["gpu"])
        vram = num(build["gpu"].get("vram")) or 0
        scores.append(min(1.0, ram / 32) * .30 + min(1.0, gpu / 350) * .40 + min(1.0, vram / 16) * .30)
    if req.get("general"):
        price = build["cpu"]["price"] + build["gpu"]["price"]
        ram = build["ram"].get("capacity") or 0
        scores.append(min(1.0, ram / 16) * .45 + (0.65 if price < 16000 else .45) * .30 + min(1.0, (build["storage"].get("capacity") or 0) / 1000) * .25)
    if not scores:
        # Budget-only requests use a neutral Balanced profile rather than
        # accidentally spending the budget on one oversized component.
        ram = build["ram"].get("capacity") or 0
        storage = build["storage"].get("capacity") or 0
        return float(0.55 * _balance_score(build) + 0.25 * min(1.0, ram / 16.0) + 0.20 * min(1.0, storage / 1000.0))
    return float(np.mean(scores))

def requirement_score(build, req):
    score = 0.0
    weights = 0.0
    total = sum(x["price"] for x in build.values() if isinstance(x, dict) and "price" in x)
    budget = req.get("budget")
    if budget:
        ratio = total / budget
        budget_fit = _budget_utilization_score(total, budget)
        score += 1.8 * budget_fit
        weights += 1.8

    usage = _usage_score(build, req)
    score += 1.5 * usage
    weights += 1.5

    # Focus priorities are explicit user preferences and can coexist.
    focus_weights = {
        "focus_cpu": _cpu_performance_score(build["cpu"]) / 100.0,
        "focus_gpu": _gpu_performance_score(build["gpu"]) / 100.0,
        "focus_ram": min(1.0, (build["ram"].get("capacity") or 0) / 32.0),
        "focus_ssd": (min(1.0, (build["storage"].get("capacity") or 0) / 1000.0) if any(x in build["storage"].get("type", "").upper() for x in ["SSD", "M.2", "NVME"]) else 0.0),
        "focus_hdd": (min(1.0, (build["storage"].get("capacity") or 0) / 2000.0) if "HDD" in build["storage"].get("type", "").upper() else 0.0),
    }
    for key, val in focus_weights.items():
        if req.get(key):
            score += 1.8 * val
            weights += 1.8

    if req.get("gaming") or req.get("focus_gpu"):
        # Gaming / GPU-focused builds need a meaningful discrete GPU + balanced CPU.
        if not build["gpu"].get("integrated"):
            score += 1.0 * _balance_score(build)
            weights += 1.0
        else:
            score -= 2.0
            weights += 1.0

    if (req.get("general") or req.get("programming")) and not req.get("focus_gpu") and not req.get("gpu"):
        # General/programming should avoid spending the budget on a discrete GPU
        # unless the customer explicitly focuses on GPU or requests one.
        if build["gpu"].get("integrated"):
            score += 1.2
        else:
            score -= 1.0 + min(1.0, build["gpu"].get("price", 0) / 10000.0)
        weights += 1.2

    if req.get("gpu"):
        sim = _gpu_similarity(req["gpu"], build["gpu"].get("chipset", ""))
        score += 1.4 * sim
        weights += 1.4

    if req.get("ram_gb"):
        cap = build["ram"]["capacity"] or 0
        fit = min(1.0, cap / req["ram_gb"]) if cap >= req["ram_gb"] else -(req["ram_gb"]-cap)/req["ram_gb"]
        score += 1.1 * fit; weights += 1.1
    if req.get("ram_type"):
        score += 1.0 if build["ram"]["ddr"] == req["ram_type"] else -1.0
        weights += 1.0
    ram_ok, _, speed_headroom = _ram_mainboard_fit(build["cpu"], build["mainboard"], build["ram"])
    if ram_ok:
        score += 0.35
        if speed_headroom >= 1000:
            score += 0.05
        weights += 0.4
    else:
        score -= 1.5
        weights += 1.0
    if req.get("storage_gb"):
        cap = build["storage"]["capacity"]
        fit = min(1.0, cap / req["storage_gb"]) if cap >= req["storage_gb"] else -(req["storage_gb"]-cap)/req["storage_gb"]
        score += .9 * fit; weights += .9
    if req.get("wifi"):
        score += 1.0 if "มี" in build["mainboard"]["wifi"] else -0.5
        weights += .6
    return score / max(weights, 1e-6)

def _feature_vector(build, req):
    total = sum(x["price"] for x in build.values() if isinstance(x, dict) and "price" in x)
    compatible, _, power = compatibility(build)
    return [
        float(req.get("budget") or 0), float(total), float((req.get("budget") or total) - total),
        float(build["cpu"]["price"]), float(build["gpu"]["price"]), float(build["ram"]["price"]),
        float(build["mainboard"]["price"]), float(build["storage"]["price"]), float(build["psu"]["price"]),
        float(build["case"]["price"]), float(build["cooler"]["price"]),
        float(build["ram"]["capacity"] or 0), float(build["storage"]["capacity"] or 0),
        float(build["cpu"].get("tdp", 0)), float(build["cpu"].get("max_power", 0)),
        float(build["gpu"].get("tdp", 0)),
        float(build["psu"]["watt"]), float(power), float(compatible),
        float(1 if req.get("gaming") else 0), float(1 if req.get("programming") else 0),
        float(1 if req.get("editing") else 0), float(1 if req.get("general") else 0), float(1 if req.get("ai_ml") else 0),
        float(1 if req.get("wifi") else 0), float(req.get("ram_gb") or 0), float(req.get("storage_gb") or 0),
        float(1 if req.get("focus_cpu") else 0), float(1 if req.get("focus_gpu") else 0),
        float(1 if req.get("focus_ram") else 0), float(1 if req.get("focus_ssd") else 0),
        float(1 if req.get("focus_hdd") else 0),
        float(build["ram"].get("bus") or 0),
        float(build["mainboard"].get("ram_max_speed") or 0),
        float(build["mainboard"].get("ram_max_capacity") or 0),
        float(build["cpu"].get("ram_native_max") or 0),
    ]


def recommendation_reasons(result, req):
    b = result["build"]
    total = result["total"]
    budget = req.get("budget")
    out = []
    if budget:
        ratio = total / budget
        out.append(f"ใช้งบประมาณประมาณ {ratio*100:.0f}% ของงบที่กำหนด")
    usage = []
    if req.get("gaming"): usage.append("Gaming")
    if req.get("programming"): usage.append("เขียนโปรแกรม")
    if req.get("general"): usage.append("ใช้งานทั่วไป")
    if req.get("editing"): usage.append("ตัดต่อ/กราฟิก")
    if req.get("ai_ml"): usage.append("AI/ML")
    if usage:
        out.append("จัดสเปกให้เหมาะกับ " + " + ".join(usage))
    if req.get("gaming"):
        out.append(f"ให้ความสำคัญกับ GPU {b['gpu']['chipset']} และ CPU สำหรับงานเกม")
    if req.get("focus_cpu"):
        out.append("ให้น้ำหนักกับ CPU ตามที่ลูกค้าระบุ")
    if req.get("focus_gpu"):
        out.append("ให้น้ำหนักกับ GPU ตามที่ลูกค้าระบุ")
    if req.get("focus_ram"):
        out.append(f"ให้น้ำหนักกับ RAM ความจุ {b['ram']['capacity']:.0f}GB")
    if req.get("focus_ssd"):
        out.append("ให้น้ำหนักกับ SSD/M.2/NVMe ตามที่ลูกค้าระบุ")
    if req.get("focus_hdd"):
        out.append("ให้น้ำหนักกับ HDD ตามที่ลูกค้าระบุ")
    if req.get("programming") and (b["ram"].get("capacity") or 0) >= 32:
        out.append("RAM อย่างน้อย 32GB เหมาะกับงานพัฒนาและการเปิดหลายโปรแกรม")
    if req.get("ram_gb") and (b["ram"].get("capacity") or 0) >= req["ram_gb"]:
        out.append(f"RAM {b['ram']['capacity']:.0f}GB ตรงตามขั้นต่ำที่ต้องการ")
    if req.get("storage_gb") and (b["storage"].get("capacity") or 0) >= req["storage_gb"]:
        out.append(f"Storage {b['storage']['capacity']:.0f}GB ตรงตามขั้นต่ำที่ต้องการ")
    mb_min, mb_label = _mainboard_requirement(b["cpu"], b.get("gpu"), b.get("ram"))
    mb_score = _mainboard_quality_score(b["mainboard"])
    out.append(f"Mainboard ระดับ {mb_score:.0f}/100 เหมาะกับ CPU; ขั้นต่ำที่ระบบกำหนด {mb_min:.0f}/100")
    ram_ok, ram_notes, _ = _ram_mainboard_fit(b["cpu"], b["mainboard"], b["ram"])
    if ram_notes:
        out.append("RAM " + " / ".join(ram_notes[:2]))
    required = result.get("estimated_power", 0)
    raw_load = result.get("estimated_load", _estimated_raw_power_watt(b))
    out.append(f"CPU TDP {b['cpu'].get('tdp', 0):.0f}W / Max {b['cpu'].get('max_power', 0):.0f}W + GPU {b['gpu'].get('tdp', 0):.0f}W")
    headroom = b['psu']['watt'] - raw_load
    out.append(f"ประเมินโหลดระบบประมาณ {raw_load:.0f}W; PSU {b['psu']['watt']:.0f}W; Headroom ประมาณ {headroom:.0f}W")
    return out[:5]

def train_ranker():
    """Train an ML regressor on synthetic prototype requirements/builds.

    Labels come from the transparent requirement_score + compatibility rules.
    This is a prototype model, not a claim of real JIB customer behavior.
    """
    rng = np.random.default_rng(42)
    rows, labels = [], []
    pools = PRODUCTS
    for _ in range(1800):
        req = {
            "budget": int(rng.choice([20000, 25000, 30000, 35000, 40000, 50000, 60000])),
            "gaming": bool(rng.integers(0, 2)),
            "programming": bool(rng.integers(0, 2)),
            "ram_gb": int(rng.choice([16, 32, 64])),
            "ram_type": rng.choice(["DDR4", "DDR5"]),
            "storage_gb": int(rng.choice([500, 1000, 2000])),
        }
        if rng.random() < .45:
            req["gpu"] = rng.choice(["RTX 4060", "RTX 4060 TI", "RTX 4070", "RTX 5070"])
        build = {
            "cpu": rng.choice(pools["cpu"]), "gpu": rng.choice(pools["gpu"]), "ram": rng.choice(pools["ram"]),
            "mainboard": rng.choice(pools["mainboard"]), "storage": rng.choice(pools["storage"]),
            "case": rng.choice(pools["case"]), "psu": rng.choice(pools["psu"]), "cooler": rng.choice(pools["cooler"])
        }
        rows.append(_feature_vector(build, req))
        compatible, _, _ = compatibility(build)
        label = requirement_score(build, req) + (1.2 if compatible else -1.5)
        labels.append(label)

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=float)
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=250, max_depth=6, learning_rate=.06, subsample=.85,
                             colsample_bytree=.85, objective="reg:squarederror", random_state=42)
        model_type = "XGBRegressor"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model_type = "RandomForestRegressor"
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "model_type": model_type}, f)
    return model_type


def _gpu_vram_gb(gpu):
    """Parse GPU VRAM capacity in GB from the normalized GPU record."""
    raw = clean_text(gpu.get("vram"))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GB", raw, flags=re.I)
    return float(m.group(1)) if m else 0.0


def _is_nvidia_gpu(gpu):
    """Identify NVIDIA discrete GPUs for workload-aware AI/ML ranking."""
    if gpu.get("integrated"):
        return False
    text = (clean_text(gpu.get("chipset")) + " " + clean_text(gpu.get("name"))).upper()
    return "NVIDIA" in text or "RTX" in text or "GEFORCE" in text


def _gpu_model_token(s):
    s = clean_text(s).upper()
    m = re.search(r"(?:RTX|RX)\s*\d{3,4}(?:\s*(?:TI|SUPER|XT|XTX))?", s)
    return re.sub(r"[^A-Z0-9]", "", m.group(0)) if m else ""


def _gpu_matches(target, actual):
    t = _gpu_model_token(target) or re.sub(r"[^A-Z0-9]", "", clean_text(target).upper())
    a = _gpu_model_token(actual) or re.sub(r"[^A-Z0-9]", "", clean_text(actual).upper())
    return a == t


def _cpu_model_parts(s):
    t = clean_text(s).upper()
    # Capture Intel Core / Core Ultra / AMD Ryzen family + model number + suffix.
    m = re.search(r"(?:CORE\s+)?(I[3579]|ULTRA\s*[3579]|RYZEN\s*[3579])\s*[- ]?(\d{4,5})([A-Z0-9]*)", t)
    if not m:
        return "", "", ""
    family = re.sub(r"[^A-Z0-9]", "", m.group(1))
    number = m.group(2)
    suffix = re.sub(r"[^A-Z0-9]", "", m.group(3) or "")
    return family, number, suffix


def _cpu_model_token(s):
    family, number, suffix = _cpu_model_parts(s)
    return family + number + suffix if family else re.sub(r"[^A-Z0-9]", "", clean_text(s).upper())


def _cpu_matches(target, actual):
    tf, tn, ts = _cpu_model_parts(target)
    af, an, ass = _cpu_model_parts(actual)
    if not tf or not af or tf != af or tn != an:
        return False
    # If the user specified a suffix (F/KF/X3D/etc.), require that exact suffix.
    # Without a suffix, prefer the exact base model rather than silently switching
    # to F/KF/other variants.
    return ts == ass


def _gpu_family_number(token):
    token = clean_text(token).upper().replace(" ", "")
    m = re.search(r"(RTX|RX)(\d{3,4})", token)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _gpu_similarity(target, actual):
    """Return a transparent 0..1 similarity for fallback GPU recommendations."""
    tf = _gpu_family_number(target)
    af = _gpu_family_number(actual)
    if not tf or not af or tf[0] != af[0]:
        return 0.0
    diff = abs(tf[1] - af[1])
    # Model numbers are not a strict performance scale. This similarity is
    # only used to surface transparent fallback options from the same family.
    return max(0.0, 1.0 - diff / 1200.0)


def _gpu_fallback_candidates(target, pool, limit=6):
    ranked = []
    for x in pool:
        sim = _gpu_similarity(target, x.get("chipset", ""))
        if sim > 0:
            ranked.append((sim, x.get("price", float("inf")), x))
    ranked.sort(key=lambda z: (-z[0], z[1]))
    return [x[2] for x in ranked[:limit]]


def _candidate_filter(req):
    pools = {k: list(v) for k, v in PRODUCTS.items()}
    if req.get("cpu_model"):
        exact_cpu = [x for x in pools["cpu"] if _cpu_matches(req["cpu_model"], x["name"]) ]
        if exact_cpu:
            pools["cpu"] = exact_cpu
        else:
            # Do not silently replace an explicitly requested CPU. An empty CPU pool
            # will make the build search report that the requested model is unavailable.
            pools["cpu"] = []
    if req.get("gpu"):
        exact = [x for x in pools["gpu"] if _gpu_matches(req["gpu"], x["chipset"]) or _gpu_matches(req["gpu"], x["name"])]
        if exact:
            pools["gpu"] = exact
        else:
            # Do not silently replace the customer's requested GPU.
            # If it is absent from the source CSV, keep a small transparent
            # set of nearby same-family GPUs so the system can offer alternatives.
            pools["gpu"] = _gpu_fallback_candidates(req["gpu"], pools["gpu"], limit=6)
    # AI/ML workload guardrail: when the user has not explicitly requested a GPU,
    # prefer NVIDIA discrete GPUs in this prototype and favor higher VRAM when the
    # budget permits. This prevents the generic GPU score from selecting a Radeon
    # alternative for an AI/ML request simply because its raw performance score is high.
    if req.get("ai_ml") and not req.get("gpu"):
        nvidia = [x for x in pools["gpu"] if _is_nvidia_gpu(x)]
        if nvidia:
            budget = req.get("budget") or float("inf")
            # Keep candidates that can realistically fit into a complete build.
            affordable_nvidia = [x for x in nvidia if x.get("price", 0) <= budget * 0.60]
            pools["gpu"] = affordable_nvidia or nvidia

    if req.get("ram_type"):
        pools["ram"] = [x for x in pools["ram"] if x["ddr"] == req["ram_type"]]
    if req.get("ram_gb"):
        pools["ram"] = [x for x in pools["ram"] if (x["capacity"] or 0) >= req["ram_gb"]]
    else:
        # A complete modern PC should not fall back to 8GB RAM just to spend the budget elsewhere.
        baseline_ram = [x for x in pools["ram"] if (x.get("capacity") or 0) >= 16]
        if baseline_ram:
            pools["ram"] = baseline_ram
    if req.get("storage_gb"):
        pools["storage"] = [x for x in pools["storage"] if x["capacity"] >= req["storage_gb"]]
    if req.get("focus_hdd"):
        hdd = [x for x in pools["storage"] if "HDD" in x["type"].upper()]
        preferred = [x for x in hdd if (x.get("capacity") or 0) >= 1000]
        if preferred:
            hdd = preferred
        if hdd:
            pools["storage"] = hdd
    elif req.get("storage_type") == "SSD" or req.get("focus_ssd") or req.get("gaming") or req.get("programming") or req.get("editing") or req.get("ai_ml") or not any(req.get(k) for k in ["focus_hdd"]):
        # Default complete-PC storage is SSD/NVMe. HDD is selected explicitly with "เน้น HDD".
        ssd = [x for x in pools["storage"] if any(t in x["type"].upper() for t in ["SSD", "M.2", "NVME"]) or "SSD" in x["name"].upper() or "M.2" in x["name"].upper()]
        if req.get("focus_ssd"):
            preferred = [x for x in ssd if (x.get("capacity") or 0) >= 500]
            if preferred:
                ssd = preferred
        elif not req.get("storage_gb"):
            preferred = [x for x in ssd if (x.get("capacity") or 0) >= 500]
            if preferred:
                ssd = preferred
        if ssd:
            pools["storage"] = ssd
    if req.get("wifi"):
        wifi = [x for x in pools["mainboard"] if "มี" in x["wifi"]]
        if wifi:
            pools["mainboard"] = wifi

    # General-use builds should not automatically buy a discrete GPU.
    # Keep only CPUs with integrated graphics and use a zero-cost virtual GPU entry.
    if (req.get("general") or req.get("programming")) and not req.get("focus_gpu") and not req.get("gpu") and not req.get("gaming") and not req.get("editing") and not req.get("ai_ml"):
        # General/programming builds do not need a discrete GPU by default.
        # Prefer an iGPU-capable CPU so budget can go toward CPU/RAM/storage.
        igpu_cpus = [x for x in pools["cpu"] if _cpu_has_integrated_graphics(x)]
        if igpu_cpus:
            pools["cpu"] = igpu_cpus
            pools["gpu"] = [_integrated_gpu()]

    # Focused component pools: keep the search centered on the component the
    # customer explicitly prioritized instead of letting the ML/budget score
    # collapse everything back to cheap baseline parts.
    if req.get("focus_gpu") and not req.get("gpu"):
        budget = req.get("budget") or float("inf")
        feasible_gpu = [x for x in pools["gpu"] if x.get("price", 0) <= budget * 0.55]
        if feasible_gpu:
            pools["gpu"] = feasible_gpu
        ranked_gpu = sorted(pools["gpu"], key=lambda x: _gpu_performance_score(x), reverse=True)
        cheap_gpu = sorted(pools["gpu"], key=lambda x: x.get("price", float("inf")))
        merged_gpu = ranked_gpu[:12] + cheap_gpu[:18]
        seen_gpu = set(); pools["gpu"] = []
        for x in merged_gpu:
            if x["name"] not in seen_gpu:
                seen_gpu.add(x["name"]); pools["gpu"].append(x)
    if req.get("focus_cpu"):
        budget = req.get("budget") or float("inf")
        cpu_cap = _usage_budget_policy(req)["cpu_cap"] if budget != float("inf") else 0.42
        feasible_cpu = [x for x in pools["cpu"] if x.get("price", 0) <= budget * cpu_cap]
        if feasible_cpu:
            pools["cpu"] = feasible_cpu
        # CPU focus should not fall back to entry-level CPUs. Keep a sensible
        # performance floor that scales with the budget.
        cpu_floor = 60.0 if budget >= 35000 else (52.0 if budget >= 25000 else 48.0)
        focused_cpu = [x for x in pools["cpu"] if _cpu_performance_score(x) >= cpu_floor]
        if focused_cpu:
            pools["cpu"] = focused_cpu
        ranked_cpu = sorted(pools["cpu"], key=lambda x: _cpu_performance_score(x), reverse=True)
        cheap_cpu = sorted(pools["cpu"], key=lambda x: x.get("price", float("inf")))
        merged_cpu = ranked_cpu[:16] + cheap_cpu[:16]
        seen_cpu = set(); pools["cpu"] = []
        for x in merged_cpu:
            if x["name"] not in seen_cpu:
                seen_cpu.add(x["name"]); pools["cpu"].append(x)
    if req.get("focus_cpu") and not req.get("focus_gpu") and not req.get("gpu"):
        budget = req.get("budget") or float("inf")
        feasible_gpu = [x for x in pools["gpu"] if x.get("price", 0) <= budget * 0.35]
        if feasible_gpu:
            pools["gpu"] = feasible_gpu

    if req.get("focus_ram"):
        budget = req.get("budget") or float("inf")
        feasible_ram = [x for x in pools["ram"] if x.get("price", 0) <= budget * 0.25]
        if feasible_ram:
            pools["ram"] = feasible_ram
        ranked_ram = sorted(pools["ram"], key=lambda x: ((x.get("capacity") or 0), -(x.get("price") or 0)), reverse=True)
        pools["ram"] = ranked_ram[:18] if len(ranked_ram) > 18 else ranked_ram

    # Storage defaults: 512GB minimum; gaming on a healthy budget targets 1TB.
    # Only an explicit focus on storage is allowed to spend disproportionately more.
    if not req.get("storage_gb") and not req.get("focus_ssd") and not req.get("focus_hdd"):
        target = 1000 if req.get("gaming") and (req.get("budget") or 0) >= 35000 else 500
        ssd = [x for x in pools["storage"] if (x.get("capacity") or 0) >= target and any(t in x.get("type", "").upper() for t in ["SSD", "M.2", "NVME"])]
        if ssd:
            # For non-storage-focused builds, exclude unusually expensive drives
            # that would consume an unreasonable share of the whole budget.
            budget = req.get("budget") or 0
            if budget:
                reasonable = [x for x in ssd if x.get("price", 0) <= budget * 0.15]
                if reasonable:
                    ssd = reasonable
            pools["storage"] = ssd

    return pools


def _shortlist(pool, req, key, n=12):
    def s(x):
        val = 0.0
        p = x.get("price", 0)
        if key == "gpu":
            if req.get("focus_gpu"):
                val += _gpu_performance_score(x) / 12.0
            if req.get("gpu"):
                val += 20 if _gpu_matches(req["gpu"], x.get("chipset", "")) else 0
            if req.get("gaming") or req.get("editing") or req.get("ai_ml"):
                val += _gpu_power_estimate(x) / 100.0
            if req.get("ai_ml"):
                # Transparent prototype preference: NVIDIA + larger VRAM gets
                # additional weight for AI/ML workloads.
                val += 6.0 if _is_nvidia_gpu(x) else -5.0
                val += min(1.0, _gpu_vram_gb(x) / 16.0) * 4.0
            elif req.get("general"):
                if x.get("integrated"):
                    val += 8.0
                else:
                    val += -4.0 - min(p / 5000, 6.0)
            else:
                val += -p * 0.00001
        elif key == "cpu":
            if req.get("focus_cpu"):
                val += _cpu_performance_score(x) / 20.0
            elif req.get("gaming") or req.get("programming") or req.get("editing") or req.get("ai_ml"):
                val += _cpu_power_estimate(x) / 120.0
            else:
                val += _cpu_performance_score(x) / 100.0 - p * 0.00001
        elif key == "mainboard":
            # Mainboard is a basic system-matching constraint, not a user focus.
            # Prefer boards that fit the CPU power/platform instead of simply
            # choosing the cheapest socket-compatible board.
            if "cpu" in req:
                pass
            cpu_hint = None
            # At shortlist time CPU may not be available in x; use dataset-level
            # platform tier and price/feature clues. Final compatibility enforces
            # the exact CPU requirement.
            q = _mainboard_quality_score(x)
            val += q / 25.0
            if req.get("focus_cpu") or req.get("gaming") or req.get("programming") or req.get("editing") or req.get("ai_ml"):
                val += q / 100.0
            val -= p * 0.00001
        elif key == "cooler":
            if req.get("focus_cpu"):
                val += _cooler_capacity(x) / 100.0
                if "liquid" in clean_text(x.get("type")).lower():
                    val += 2.0
                val -= p * 0.00001
            else:
                val += -p * 0.00001
        elif key == "ram":
            if req.get("focus_ram"):
                val += min((x.get("capacity", 0) or 0) / 32, 1.0) * 5
            elif req.get("ram_gb"):
                val += min((x.get("capacity", 0) or 0) / req["ram_gb"], 1.5) * 3
            elif req.get("programming") or req.get("editing") or req.get("ai_ml"):
                val += min((x.get("capacity", 0) or 0) / 32, 1.5) * 2
            else:
                val += min((x.get("capacity", 0) or 0) / 16, 1.2)
        elif key == "storage":
            if req.get("focus_ssd") or req.get("focus_hdd"):
                target_cap = 1000 if req.get("focus_ssd") else 2000
                val += min((x.get("capacity", 0) or 0) / target_cap, 1.2) * 3
                if req.get("focus_ssd") and any(t in x.get("type", "").upper() for t in ["SSD", "M.2", "NVME"]):
                    val += 5
                if req.get("focus_hdd") and "HDD" in x.get("type", "").upper():
                    val += 5
            elif req.get("storage_gb"):
                val += min((x.get("capacity", 0) or 0) / req["storage_gb"], 1.5) * 3
            else:
                # Default storage: 512GB minimum. Gaming with a larger budget
                # targets 1TB, but does not force an expensive PCIe 5.0 drive.
                cap=x.get("capacity",0) or 0; typ=x.get("type","").upper()
                if req.get("general"):
                    if "SATA SSD" in typ: val += 4.0
                    elif "HDD" in typ: val += 1.2
                    elif "M.2" in typ or "NVME" in typ: val += 1.0
                    target_cap = 500
                elif req.get("gaming"):
                    if "M.2" in typ or "NVME" in typ: val += 3.0
                    elif "SATA SSD" in typ: val += 2.0
                    target_cap = 1000 if (req.get("budget") or 0) >= 35000 else 500
                else:
                    if "SATA SSD" in typ: val += 2.5
                    elif "M.2" in typ or "NVME" in typ: val += 2.5
                    target_cap = 500
                val += max(0.0, 1.0 - abs(cap - target_cap) / max(target_cap, 500)) * 2.5
                # Avoid luxury storage unless the user explicitly focused storage.
                if req.get("budget") and x.get("price", 0) > req["budget"] * 0.15:
                    val -= 6.0
                val -= min(2.0, x.get("price", 0) / max(req.get("budget", 40000), 1) * 2.0)
        else:
            val += -p * 0.00001
        return val
    ranked = sorted(pool, key=s, reverse=True)
    # Keep inexpensive options alongside performance-focused options so a
    # tight budget is not eliminated before the full build is assembled.
    cheapest = sorted(pool, key=lambda x: x.get("price", float("inf")))[:min(8, len(pool))]
    merged = []
    seen = set()
    for x in ranked[:n]:
        marker = x.get("name")
        if marker not in seen:
            seen.add(marker); merged.append(x)
    for x in cheapest:
        marker = x.get("name")
        if marker not in seen:
            seen.add(marker); merged.append(x)
    return merged[:max(n, min(len(merged), n + 6))]

def recommend_builds(req, top_n=3):
    """Generate complete PC builds, then rank them with rules + ML.

    The search is intentionally component-aware instead of letting a cheap
    early beam candidate eliminate stronger combinations. This is important
    for large budgets: a 100k request must still be able to reach high-end
    CPU/GPU combinations before ranking. PSU is selected only after the whole
    build is known, so wattage and PSU price remain proportional to the build.
    """
    pools = _candidate_filter(req)
    budget = req.get("budget") or float("inf")

    # Optional per-Build edit mode. The app can lock the current build and
    # search only the component being changed. This keeps Build 1/2/3 as
    # genuinely separate alternatives instead of applying one change globally.
    locked = req.get("_locked_build") or {}
    edit_component = req.get("_edit_component")
    if locked and edit_component:
        for key in ["cpu", "gpu", "mainboard", "ram", "storage", "psu", "case", "cooler"]:
            if key == edit_component:
                continue
            item = locked.get(key)
            if not item:
                continue
            name = item.get("name") if isinstance(item, dict) else None
            if not name:
                continue
            pools[key] = [x for x in pools.get(key, []) if x.get("name") == name]
            if not pools[key]:
                return []
        # During a component edit, the exact requirement for the edited
        # component must not accidentally filter out all candidate products.
        if edit_component == "cpu":
            req.pop("cpu_model", None)
        elif edit_component == "gpu":
            req.pop("gpu", None)
        elif edit_component == "ram":
            ram_model = req.get("_ram_model")
            req.pop("ram_gb", None); req.pop("ram_type", None)
            if ram_model:
                exact_ram = [x for x in pools.get("ram", []) if x.get("name") == ram_model]
                if not exact_ram:
                    return []
                pools["ram"] = exact_ram
        elif edit_component == "storage":
            req.pop("storage_gb", None)

    def comp_rank(x, key):
        p = x.get("price", 0) or 0
        if key == "cpu":
            v = _cpu_performance_score(x)
            if req.get("focus_cpu"): v += 30
            if req.get("programming") or req.get("editing") or req.get("ai_ml"): v += 10
            if req.get("gaming"): v += 8
            return v - p / max(budget, 1) * 8
        if key == "gpu":
            v = _gpu_performance_score(x)
            if req.get("focus_gpu"): v += 30
            if req.get("gaming") or req.get("editing") or req.get("ai_ml"): v += 10
            if req.get("ai_ml"):
                v += 18 if _is_nvidia_gpu(x) else -12
                v += min(12.0, _gpu_vram_gb(x)) * 0.75
            if req.get("general") and not req.get("focus_gpu") and not req.get("gpu"):
                v = 100 if x.get("integrated") else -p / 1000
            return v - p / max(budget, 1) * 6
        if key == "ram":
            v = min(1.5, (x.get("capacity") or 0) / 32) * 10
            if req.get("focus_ram"): v += 20
            return v - p / max(budget, 1) * 5
        if key == "storage":
            cap = x.get("capacity") or 0; typ = x.get("type", "").upper()
            v = min(1.5, cap / (1000 if req.get("gaming") else 500)) * 8
            if req.get("focus_ssd") and any(t in typ for t in ["SSD", "M.2", "NVME"]): v += 20
            if req.get("focus_hdd") and "HDD" in typ: v += 20
            if req.get("general") and "SATA SSD" in typ: v += 4
            return v - p / max(budget, 1) * 8
        return -p / max(budget, 1)

    def choose(key, n_high=18, n_cheap=8):
        pool = list(pools[key])
        if not pool: return []
        ranked = sorted(pool, key=lambda x: comp_rank(x, key), reverse=True)
        cheap = sorted(pool, key=lambda x: x.get("price", float("inf")))[:n_cheap]
        out=[]; seen=set()
        for x in ranked[:n_high] + cheap:
            marker=x.get("name")
            if marker in seen: continue
            seen.add(marker); out.append(x)
        return out

    policy = _usage_budget_policy(req)
    cpus = choose("cpu", 24, 10)
    # Remove disproportionate CPU options when CPU is not explicitly focused.
    if req.get("budget"):
        cpu_allowed = [c for c in cpus if _component_price_ok(c, policy["cpu_cap"], req["budget"]) or req.get("focus_cpu")]
        if cpu_allowed:
            cpus = cpu_allowed
    gpus = choose("gpu", 24, 10)
    if req.get("general") and not req.get("focus_gpu") and not req.get("gpu"):
        gpus = [_integrated_gpu()]
    cpu_floor = _minimum_cpu_performance_for_usage(req, gpus)
    if cpu_floor > 0:
        stronger_cpus = [c for c in cpus if _cpu_performance_score(c) >= cpu_floor]
        if stronger_cpus:
            cpus = stronger_cpus

    # RAM value strategy. For a requested capacity such as 32GB, prefer the
    # cheapest compatible kit in the requested capacity and keep luxury kits out
    # unless RAM is explicitly the focus. This also makes DDR4 a valid value path
    # when no DDR5 requirement was stated.
    def ram_value_score(x):
        cap = x.get("capacity") or 0
        p = x.get("price") or 0
        score = 0.0
        if req.get("ram_gb"):
            score += 18.0 if cap >= req["ram_gb"] else -8.0
            if cap == req["ram_gb"]:
                score += 3.0
        elif req.get("focus_ram"):
            score += min(1.0, cap / 32.0) * 14.0
        elif req.get("programming") or req.get("editing") or req.get("ai_ml"):
            target = 32 if (req.get("budget") or 0) >= 35000 else 16
            score += 6.0 if cap >= target else -3.0
        elif req.get("gaming"):
            target = 32 if (req.get("budget") or 0) >= 60000 else 16
            score += 5.0 if cap >= target else -3.0
        else:
            score += 5.0 if cap >= 16 else 0.0
        price_per_gb = p / max(cap, 1)
        # For non-focused RAM, price/value is a first-class criterion. The
        # system should not pick a premium RGB kit merely because it is a
        # little faster when a cheaper kit satisfies the requirement.
        if not req.get("focus_ram"):
            score -= min(16.0, price_per_gb * 0.45)
            score -= _budget_share(p, req.get("budget") or 0) * 16.0
        else:
            score -= min(10.0, price_per_gb * 0.15)
            score -= _budget_share(p, req.get("budget") or 0) * 10.0
        return score

    if req.get("ram_gb"):
        ram_target_pool = [r for r in pools["ram"] if (r.get("capacity") or 0) == req["ram_gb"]]
        affordable = [r for r in ram_target_pool if _component_price_ok(r, policy["ram_cap"], req.get("budget") or 0)]
        if affordable:
            ram_target_pool = affordable
        if not req.get("ram_type") and ram_target_pool:
            # Keep all compatible DDR types, but price/value scoring naturally
            # prefers lower total platform cost when the user did not require DDR5.
            pass
    elif req.get("focus_ram"):
        ram_target_pool = list(pools["ram"])
    elif req.get("programming") or req.get("editing") or req.get("ai_ml"):
        budget_now = req.get("budget") or 0
        has_reasonable_32 = any((r.get("capacity") or 0) >= 32 and _component_price_ok(r, policy["ram_cap"], budget_now) for r in pools["ram"]) if budget_now else False
        target = 32 if has_reasonable_32 else 16
        ram_target_pool = [r for r in pools["ram"] if (r.get("capacity") or 0) == target]
    elif req.get("gaming"):
        # In mixed-workload requests, programming/editing/AI takes precedence
        # over the Gaming-only 16GB default when 32GB is reasonably affordable.
        mixed_heavy = any(req.get(k) for k in ["programming", "editing", "ai_ml"])
        target = 32 if ((req.get("budget") or 0) >= 60000 or (mixed_heavy and (req.get("budget") or 0) >= 35000)) else 16
        ram_target_pool = [r for r in pools["ram"] if (r.get("capacity") or 0) == target]
    else:
        ram_target_pool = [r for r in pools["ram"] if (r.get("capacity") or 0) == 16]

    if ram_target_pool and req.get("budget") and not req.get("focus_ram"):
        affordable_target = [r for r in ram_target_pool if _component_price_ok(r, policy["ram_cap"], req["budget"])]
        if affordable_target:
            ram_target_pool = affordable_target
    if ram_target_pool:
        pools["ram"] = ram_target_pool
    rams = sorted(pools["ram"], key=ram_value_score, reverse=True)[:14]
    storages = choose("storage", 6, 3)
    model_pack = None
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f: model_pack = pickle.load(f)
        except Exception: model_pack = None

    # High-level component pair search. Matching platform parts are selected
    # after CPU/GPU, so mainboard quality is a real constraint rather than a
    # cosmetic preference.
    candidates=[]
    for cpu in cpus:
        for gpu in gpus:
            if req.get("gpu") and not (_gpu_matches(req["gpu"], gpu.get("chipset", "")) or _gpu_matches(req["gpu"], gpu.get("name", ""))):
                continue
            mb_pool=[m for m in pools["mainboard"] if (not cpu["socket"] or not m["socket"] or cpu["socket"]==m["socket"])]
            mb_pool=[m for m in mb_pool if _mainboard_quality_score(m) >= _mainboard_requirement(cpu,gpu)[0]]
            if req.get("budget"):
                mb_affordable = [m for m in mb_pool if _component_price_ok(m, policy["mb_cap"], req["budget"])]
                # Keep a fallback only when the affordable pool is empty; this
                # prevents a high-end board from consuming the whole budget for
                # a normal CPU requirement.
                if mb_affordable:
                    mb_pool = mb_affordable
            mb_quality=sorted(mb_pool,key=lambda m:(-_mainboard_quality_score(m), m.get("price",0)))[:3]
            mb_cheap=sorted(mb_pool,key=lambda m:m.get("price",float("inf")))[:3]
            mb_pool=[]; mb_seen=set()
            for m in mb_quality+mb_cheap:
                if m["name"] in mb_seen: continue
                mb_seen.add(m["name"]); mb_pool.append(m)
            if not mb_pool: continue
            for mb in mb_pool[:4]:
                ram_pool=[r for r in rams if _ram_mainboard_fit(cpu,mb,r)[0]]
                if not ram_pool: continue
                # Choose RAM by workload-appropriate capacity and value, not
                # simply by largest capacity. This prevents a 9k+ 32GB kit
                # from consuming the budget when 16GB is sufficient.
                def ram_pick_score(r):
                    cap = r.get("capacity") or 0
                    p = r.get("price") or 0
                    value = 0.0
                    if req.get("ram_gb"):
                        value += 15.0 if cap >= req["ram_gb"] else -20.0
                        if cap == req["ram_gb"]:
                            value += 3.0
                    elif req.get("focus_ram"):
                        value += min(1.0, cap / 32.0) * 12.0
                    elif req.get("programming") or req.get("editing") or req.get("ai_ml"):
                        target_cap = 32 if (req.get("budget") or 0) >= 35000 else 16
                        value += 6.0 if cap >= target_cap else -3.0
                    elif req.get("gaming"):
                        target = 32 if (req.get("budget") or 0) >= 60000 else 16
                        value += 5.0 if cap >= target else -4.0
                    else:
                        value += 5.0 if cap >= 16 else 0.0
                    value -= (p / max(cap, 1)) * 0.10
                    value -= _budget_share(p, req.get("budget") or 0) * 8.0
                    if req.get("ram_gb") and _budget_share(p, req.get("budget") or 0) > policy["ram_cap"] and not req.get("focus_ram"):
                        value -= 8.0
                    return value
                ram_pool=sorted(ram_pool,key=ram_pick_score, reverse=True)[:4]
                for ram in ram_pool[:2]:
                    storage_pool=storages[:]
                    if req.get("storage_gb"):
                        storage_pool=[x for x in storage_pool if x.get("capacity",0)>=req["storage_gb"]]
                    if not storage_pool: continue
                    for storage in storage_pool[:2]:
                        # Basic lower-bound budget pruning before case/cooler.
                        base=cpu["price"]+gpu["price"]+mb["price"]+ram["price"]+storage["price"]
                        if base >= budget: continue
                        coolers=[c for c in pools["cooler"] if (not c["sockets"] or cpu["socket"] in c["sockets"]) and _cooler_is_suitable(cpu,c)]
                        if not coolers: continue
                        coolers=sorted(coolers,key=lambda c:(c.get("price",0),-_cooler_capacity(c)))[:2]
                        cases=[c for c in pools["case"] if not c["form"] or not mb["form"] or any(f.lower() in mb["form"].lower() for f in _form_factors(c["form"]))]
                        if not cases: cases=pools["case"]
                        case=min(cases,key=lambda c:c.get("price",0))
                        for cooler in coolers[:1]:
                            temp={"cpu":cpu,"gpu":gpu,"mainboard":mb,"ram":ram,"storage":storage,"case":case,"cooler":cooler}
                            psu=_select_psu_for_build(temp,budget)
                            if not psu: continue
                            b=dict(temp); b["psu"]=psu
                            total=sum(x["price"] for x in b.values())
                            if total>budget: continue
                            ok,issues,power=compatibility(b)
                            if not ok: continue
                            rule=requirement_score(b,req)
                            ml=None
                            if model_pack is not None:
                                try: ml=float(model_pack["model"].predict([_feature_vector(b,req)])[0])
                                except Exception: ml=None
                            score=(0.05*ml+0.95*rule) if ml is not None else rule
                            if req.get("budget"):
                                focus=any(req.get(k) for k in ["focus_cpu","focus_gpu","focus_ram","focus_ssd","focus_hdd"])
                                score += (7.0 if focus else 5.0)*_budget_utilization_score(total,req["budget"])
                                # Reward useful component value rather than raw spending.
                                # A build that spends 9k on RAM while using a modest
                                # CPU/GPU should lose to a build that moves that money
                                # into the main performance components.
                                ram_price = b["ram"].get("price", 0)
                                ram_cap = b["ram"].get("capacity", 0) or 0
                                ram_ratio = ram_price / max(req["budget"], 1)
                                if not req.get("focus_ram"):
                                    if ram_ratio > 0.18: score -= 3.5
                                    elif ram_ratio > 0.14: score -= 1.8
                                    elif ram_ratio > 0.10: score -= 0.6
                                if req.get("gaming") and not req.get("focus_ram") and ram_cap >= 16:
                                    score += 0.6
                            if req.get("gaming") or req.get("focus_gpu") or req.get("focus_cpu"):
                                score += 2.0*(_balance_score(b)-0.7)
                                cpu_v = _cpu_performance_score(b["cpu"]) / 100.0
                                gpu_v = _gpu_performance_score(b["gpu"]) / 100.0
                                if req.get("focus_cpu"):
                                    score += 2.2 * cpu_v
                                elif req.get("focus_gpu"):
                                    score += 2.2 * gpu_v
                                else:
                                    score += 1.2 * cpu_v + 1.8 * gpu_v
                            score += 2.0*_mainboard_fit_score(b)
                            score -= _budget_component_penalty(b, req)
                            reserve=psu["watt"]-_required_psu_watt(b)
                            if reserve>150: score-=min(4.0,(reserve-150)/40.0)
                            if reserve>=50: score+=0.25
                            raw_load = _estimated_raw_power_watt(b)
                            candidates.append({"build":b,"total":total,"score":score,"compatible":True,"issues":issues,
                                "estimated_power":power,"estimated_load":raw_load,"psu_headroom":psu["watt"]-raw_load,
                                "exact_gpu":(not req.get("gpu")) or _gpu_matches(req["gpu"],gpu.get("chipset","")) or _gpu_matches(req["gpu"],gpu.get("name","")),
                                "exact_cpu":(not req.get("cpu_model")) or _cpu_matches(req["cpu_model"],cpu.get("name","")),
                                "gpu_note":"","balance_note":_balance_note(b),"cooling_note":_cpu_cooling_requirement(cpu)["label"]})

    candidates.sort(key=lambda r:(r["score"],r["total"]),reverse=True)
    unique=[]
    seen_exact=set()
    for r in candidates:
        b=r["build"]
        exact=(b["cpu"]["name"],b["gpu"]["name"],b["mainboard"]["name"],b["ram"]["name"],b["storage"]["name"],b["psu"]["name"])
        if exact in seen_exact: continue
        # Meaningful diversity: for focused CPU+GPU requests, at least one of
        # CPU or GPU must change. For normal requests, don't duplicate the same
        # CPU/GPU pair; RAM/storage/board variants may still provide a legitimate
        # lower-cost alternative.
        # Avoid offering a very cheap third build that simply leaves most of a
        # large budget unused. Alternatives should normally stay within a useful
        # budget band unless no such compatible alternative exists.
        if unique and req.get("budget"):
            min_ratio = 0.65 if (req.get("general") and not any(req.get(k) for k in ["gaming", "programming", "editing", "ai_ml"])) else 0.72
            if r["total"] < req["budget"] * min_ratio:
                continue
        duplicate_core=False
        for u in unique:
            ub=u["build"]
            same_cpu=b["cpu"]["name"]==ub["cpu"]["name"]
            b_gpu_token = _gpu_model_token(b["gpu"].get("chipset", "")) or _gpu_model_token(b["gpu"].get("name", "")) or b["gpu"]["name"].upper()
            u_gpu_token = _gpu_model_token(ub["gpu"].get("chipset", "")) or _gpu_model_token(ub["gpu"].get("name", "")) or ub["gpu"]["name"].upper()
            same_gpu = b_gpu_token == u_gpu_token
            cpu_gap=abs(_cpu_performance_score(b["cpu"])-_cpu_performance_score(ub["cpu"]))
            gpu_gap=abs(_gpu_performance_score(b["gpu"])-_gpu_performance_score(ub["gpu"]))
            # A meaningful alternative should change the CPU/GPU performance
            # direction, not just the motherboard brand or storage model.
            if same_cpu and same_gpu:
                # General-use alternatives can legitimately differ by board or
                # storage/RAM value even when the CPU/GPU stays the same.
                general_only = req.get("general") and not any(req.get(k) for k in ["gaming", "programming", "editing", "ai_ml"])
                if not general_only:
                    duplicate_core=True; break
                same_value_parts = (
                    b["mainboard"]["name"] == ub["mainboard"]["name"] and
                    b["ram"]["name"] == ub["ram"]["name"] and
                    b["storage"]["name"] == ub["storage"]["name"]
                )
                if same_value_parts:
                    duplicate_core=True; break
            if same_cpu and same_gpu:
                # Same CPU + same GPU model is not a meaningful alternative.
                # Different board/brand/storage alone should not consume one of
                # the three ranked build slots.
                duplicate_core=True; break
            if same_cpu and gpu_gap <= 6:
                # If the GPU model is genuinely different but very close in the
                # prototype performance score, allow it only when the price
                # difference is material enough to represent a value trade-off.
                gpu_price_gap = abs((b["gpu"].get("price", 0) or 0) - (ub["gpu"].get("price", 0) or 0)) / max((ub["gpu"].get("price", 0) or 1), 1)
                if gpu_price_gap < 0.12:
                    duplicate_core=True; break
            if same_gpu and cpu_gap <= 6:
                # Do not discard a different CPU model merely because the
                # prototype performance score is identical. If price differs
                # materially, it creates a useful value-tier alternative.
                cpu_price_gap = abs((b["cpu"].get("price", 0) or 0) - (ub["cpu"].get("price", 0) or 0)) / max((ub["cpu"].get("price", 0) or 1), 1)
                if b["cpu"].get("name") == ub["cpu"].get("name") or cpu_price_gap < 0.12:
                    duplicate_core=True; break
        if duplicate_core: continue
        seen_exact.add(exact); unique.append(r)
        if len(unique)>=top_n: break

    # If the strict diversity rules leave fewer than the requested number of
    # alternatives, fill the remaining slots with the next compatible builds.
    # The UI promises Build 1/2/3 when three valid alternatives exist, so we
    # should not silently drop Build 3 merely because two candidates share a
    # CPU/GPU pair. These fallback alternatives still remain under budget and
    # are unique at the full-component level.
    if len(unique) < top_n:
        # Fallback in two passes: first seek a different CPU/GPU tier, then use
        # a different value component only when no distinct core build exists.
        remaining = [r for r in candidates if (not req.get("budget") or r["total"] <= req["budget"]) ]
        remaining.sort(key=lambda r: r["score"], reverse=True)
        for pass_mode in ("core", "value"):
            for r in remaining:
                if len(unique) >= top_n:
                    break
                b=r["build"]
                exact=(b["cpu"]["name"],b["gpu"]["name"],b["mainboard"]["name"],b["ram"]["name"],b["storage"]["name"],b["psu"]["name"])
                if exact in seen_exact:
                    continue
                is_core_distinct = True
                for u in unique:
                    ub=u["build"]
                    same_cpu = b["cpu"]["name"] == ub["cpu"]["name"]
                    b_gpu_token = _gpu_model_token(b["gpu"].get("chipset", "")) or _gpu_model_token(b["gpu"].get("name", "")) or b["gpu"]["name"].upper()
                    u_gpu_token = _gpu_model_token(ub["gpu"].get("chipset", "")) or _gpu_model_token(ub["gpu"].get("name", "")) or ub["gpu"]["name"].upper()
                    same_gpu = b_gpu_token == u_gpu_token
                    if same_cpu and same_gpu:
                        is_core_distinct = False
                        break
                if pass_mode == "core" and not is_core_distinct:
                    continue
                seen_exact.add(exact)
                unique.append(r)
    return unique[:top_n]


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("Training ranker...")
        print(train_ranker())
    req = parse_requirement(input("Requirement: "))
    print(req)
    for i, r in enumerate(recommend_builds(req), 1):
        print(i, r["total"], r["score"], r["compatible"])
        for k,v in r["build"].items(): print(" ", k, v["name"], v["price"])
