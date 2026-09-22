JIB AI Recommendation Engine V3

แนวคิด:
- JIB เป็นกรณีศึกษา
- ลูกค้าเป็นผู้มี Requirement
- AI Prototype เป็นระบบที่กลุ่มพัฒนา
- ระบบแนะนำเป็น "ชุด PC ทั้งชุด" ไม่ใช่แนะนำทีละชิ้น

ความสามารถ:
1. รับงบประมาณ/การใช้งาน/Requirement แบบต่อเนื่อง
2. สร้าง Candidate PC Builds จาก Dataset
3. ตรวจ Compatibility: CPU-Mainboard, RAM-Mainboard, Mainboard-Case, CPU-Cooler, PSU
4. ใช้ ML Ranker จัดอันดับ Build
5. แสดง Top 3 Builds พร้อมราคา คะแนน และเหตุผล/ปัญหาความเข้ากันได้

หมายเหตุสำคัญ:
- Dataset เป็นข้อมูลที่เตรียมสำหรับ Prototype ไม่ใช่ข้อมูลธุรกรรมลูกค้า JIB จริง
- ML training label ใน train_model.py สร้างจากกฎ Requirement + Compatibility เพื่อใช้ทดลองระบบ ranking; ไม่ควรอ้างว่าเป็นพฤติกรรมลูกค้าจริง
- ถ้าข้อมูลใดไม่มี field สำหรับตรวจ เช่น GPU length หรือ M.2 slot ระบบจะไม่อ้างว่าตรวจ field นั้น

วิธีรัน:
1. เปิด Command Prompt ในโฟลเดอร์นี้
2. pip install -r requirements.txt
3. python train_model.py
4. python -m streamlit run app.py

ตัวอย่าง:
งบ 30000 เล่นเกมและเขียนโปรแกรม ขอ RTX 4060 RAM 32GB DDR5 SSD 1TB

UPDATED POWER / COOLING / STORAGE LOGIC
- CPU dataset: JIB_Top50_CPU_with_TDP.csv
  - TDP (W)
  - Max Power (W) for Intel Maximum Turbo Power / AMD PPT-style peak package power used by this prototype
- GPU dataset: JIB_GPU_Top_50_with_TDP.csv
  - TDP (W) / board-power reference used by the prototype
- PSU calculation uses CPU Max Power + GPU TDP + system load + 35% headroom, then rounds upward.
- The recommender prefers a PSU with useful reserve and penalizes unnecessarily oversized PSUs.
- CPU cooling is selected from CPU TDP/Max Power and cooler type/capacity.
- Very-high-power CPU classes require closed-loop liquid cooling in this prototype.
- General-use storage favors affordable SATA SSD; M.2/NVMe receives more weight for workloads where its speed is useful.
- The ML model is retrained after the feature vector changes.

LATEST FLOW / LOGIC UPDATE
- A new budget always starts a new Requirement request and clears previous usage/specs.
- Budget + Focus (e.g. "งบ 40000 เน้น CPU") keeps the focus but asks for usage before building.
- The system does not recommend a Balanced build before usage is known.
- General-use builds prefer CPU integrated graphics when available and avoid automatically buying a discrete GPU.
- Default storage is at least 512GB; Gaming with a budget >= 35,000 targets around 1TB unless storage is explicitly focused.
- Storage-focused requests can prioritize SSD/M.2/NVMe or HDD.
- PSU target includes conservative headroom; ranking prefers useful reserve and avoids unnecessary oversizing.
- Incompatible builds are not returned as recommendations.
- XGBRegressor remains the ML scoring model; compatibility and transparent rules remain the hard constraints.


[Update - Mainboard Fit]
The recommendation engine now evaluates motherboard platform suitability as a basic compatibility rule.
Socket and DDR matching alone are not enough for high-power CPUs. The engine assigns a prototype motherboard tier
and rejects entry-level boards when the CPU power class requires a stronger platform (for example, high-power
Core i7/i9 builds will not be paired with H610-class boards). Mainboard Fit is shown in the Streamlit result.

The engine also keeps the existing Fresh Requirement / Focus / Storage / PSU logic:
- New budget starts a fresh request.
- Budget + focus asks for usage before building.
- Storage uses 512GB as a normal minimum and targets 1TB for higher-budget gaming.
- PSU uses conservative reserve.
- General-use builds only use Integrated Graphics when the selected CPU actually has an iGPU.


--- Mainboard RAM Compatibility Update ---
เพิ่มการตรวจสอบความสัมพันธ์ระหว่าง CPU + Mainboard + RAM:
- DDR4/DDR5 ต้องตรงกัน
- RAM Max Speed ของ Mainboard
- RAM Max Capacity ของ Mainboard
- XMP / EXPO profile ที่ Mainboard รองรับ
- CPU Native RAM Speed เป็น baseline; หาก RAM สูงกว่า Native แต่ Mainboard และ profile รองรับ จะถือเป็น Memory OC ไม่ใช่ incompatibility อัตโนมัติ
- ระบบแสดง RAM Compatibility และ RAM Analysis ใน UI
หมายเหตุ: ค่า Max/OC เป็นค่าอ้างอิงสำหรับ prototype; ความเร็วจริงขึ้นกับ CPU IMC, จำนวน DIMM, rank, BIOS และ Memory QVL ของผู้ผลิต

--- Budget Optimization / Component Value Update ---
- Non-focus RAM is no longer selected by maximum capacity alone.
- Gaming under 60,000 THB uses 16GB as the normal prototype baseline; 32GB is considered for higher budgets or explicit RAM focus.
- Programming/editing/AI workloads can use 32GB when a compatible 32GB kit is reasonably priced; otherwise the system keeps a value-oriented 16GB baseline.
- RAM price share is penalized when it consumes an unreasonable portion of the budget without an explicit RAM focus.
- The ranking gives more weight to useful CPU/GPU performance in Gaming/focused builds so leftover budget can support core performance rather than luxury RAM/PSU.
- General/programming builds without GPU focus or an explicit GPU request prefer CPUs with integrated graphics and do not automatically buy a discrete GPU.
- PSU sizing now derives the target from total estimated system load + headroom and GPU power class; there is no blanket 750/850W floor merely because the CPU is high-end.
- Build diversity still requires meaningful CPU/GPU differences and avoids returning duplicate core builds.
- These rules are prototype heuristics based on the project dataset, not real JIB customer behavior or benchmark guarantees.
