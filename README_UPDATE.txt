JIB AI Recommendation Engine V3 - Consolidated Fixes

รอบนี้เป็นการรวมแก้ปัญหาที่พบจากการรีเทส โดยรักษาโครงสร้างเดิมและฟีเจอร์ที่ผ่านการทดสอบไว้

1. Usage Normalization ไทย + English + Mixed
- รองรับ Gaming / เล่นเกม / เกม / เกมมิ่ง
- รองรับ Programming / Coding / Development / Developer / เขียนโปรแกรม / เขียนโค้ด / โค้ด
- รองรับ Editing / Edit / Video Editing / Photo Editing / Graphics / ตัดต่อ / แต่งภาพ / กราฟิก
- รองรับ General Use / General / Office / Work / ใช้งานทั่วไป / งานทั่วไป / ออฟฟิศ / เอกสาร
- รองรับ AI / AI-ML / Artificial Intelligence / Machine Learning / Deep Learning / ปัญญาประดิษฐ์
- หนึ่งข้อความสามารถระบุหลายการใช้งานพร้อมกันได้ เช่น
  งบ 40000 Gaming Programming Editing
  งบ 40000 เล่นเกม เขียนโปรแกรม ตัดต่อ

2. Budget reset behavior
- ทุกข้อความที่เริ่มด้วยงบใหม่จะเริ่ม Requirement รอบใหม่
- จะไม่ลาก Usage เก่ามาปนกับงบใหม่
- งบ + Focus โดยยังไม่มี Usage จะถาม Usage ใหม่ก่อนจัดชุด

3. Budget allocation / Component value
- เพิ่ม guardrail สัดส่วนงบของ CPU, Mainboard และ RAM
- ลดกรณี CPU ตัวเดียวกินงบมากเกินไปเมื่อไม่ได้ระบุ Focus CPU
- Programming / Editing / AI มีการจัด RAM 32GB เมื่ออยู่ในงบและมีตัวเลือกที่สมเหตุสมผล
- General Use เน้น CPU/RAM/Storage ให้คุ้มค่า โดยไม่ซื้อ GPU แยกโดยอัตโนมัติ
- Storage ทั่วไปยังคงขั้นต่ำประมาณ 512GB และ Gaming งบสูงมีแนวโน้ม 1TB

4. RAM value improvement
- เมื่อผู้ใช้ระบุ RAM 32GB ระบบจะเลือกชุด 32GB ที่คุ้มค่ากว่า ไม่เลือกชุดแพงสุดโดยอัตโนมัติ
- ถ้าไม่ได้ระบุ DDR5 ระบบสามารถเลือก DDR4 เพื่อประหยัดงบเมื่อเข้ากันได้
- ฟีเจอร์ conversational edit เดิมยังคงอยู่:
  เพิ่ม RAM -> ถามขนาด -> 32GB / 64GB -> จัดชุดใหม่
  เพิ่ม RAM เป็น 32GB -> ปรับตรงได้ทันที

5. CPU/GPU basic balance
- เพิ่ม CPU performance floor สำหรับ Gaming / Editing / AI และ GPU-focused builds
- ลดกรณี CPU ระดับเริ่มต้นจับคู่กับ GPU ระดับสูงเกินไป

6. Mainboard / RAM compatibility
- ยังคงตรวจ Socket, DDR, Mainboard RAM speed ceiling, capacity, XMP/EXPO และ CPU native RAM speed
- Mainboard suitability ยังคงเป็นกฎพื้นฐานของแพลตฟอร์ม ไม่ใช่ User Focus

7. PSU analysis
- ยังคงเลือก PSU ให้มีสำรองโดยไม่ตั้งใจให้ใหญ่เกินความจำเป็น
- UI เปลี่ยนเป็น:
  Estimated Load / Recommended PSU / Headroom
- แยกค่าประเมินโหลดจริงโดยประมาณออกจากค่า PSU target ที่มี reserve แล้ว

8. Build diversity
- พยายามหลีกเลี่ยง Build 1/2/3 ที่เป็นชุดแกนเดียวกันเกินไป
- General Use อนุญาตทางเลือกที่ต่างกันด้าน Mainboard / RAM / Storage ได้ แม้ CPU/GPU แกนเดียวกัน

9. ML
- ยังคงใช้ XGBRegressor
- โมเดลเป็นส่วนหนึ่งของระบบ Hybrid ที่ทำงานร่วมกับ Rule + Compatibility
- Training label เป็น synthetic/rule-derived สำหรับ Prototype ไม่ใช่ accuracy จากข้อมูลลูกค้าจริง

10. Scope ที่ยังไม่ได้เพิ่มในรอบนี้
- Conversational edit หลังแนะนำ Build ยังคงรองรับเฉพาะ "เพิ่ม RAM" ตามที่ทดสอบไว้
- ยังไม่ได้เพิ่มคำสั่งแก้ SSD / CPU / GPU / HDD เพื่อไม่ให้รอบนี้ปน Feature ใหม่

Run:
    python train_model.py
    streamlit run app.py
