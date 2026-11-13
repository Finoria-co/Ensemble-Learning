# แบ่งงาน Classwork Ensemble (ส่ง 12:30 น. 2026-11-13)

แบ่งงานตามอัลกอริทึม แต่ละคนต้องอธิบายส่วนของตัวเองให้อาจารย์ฟังได้
สถานะ: `[ ]` ยังไม่ทำ · `[~]` กำลังทำ · `[x]` เสร็จ (อัปเดตในบรรทัดของตัวเองตอนเปิด PR)

## ปฏิมา ถิ่นสำราญ (6752300461): Bagging + ดูแล repo · branch `patima/...`
- [x] `BaggingClassifier`: bootstrap ด้วย NumPy + เทรนขนานด้วย `ProcessPoolExecutor` (initializer + chunksize)
- [x] `majority_vote` (ใช้ร่วมกับ AdaBoost)
- [x] notebook ฉบับแรก: กราฟ accuracy, เวลา parallel vs sequential, confusion matrix
- [x] README, CLAUDE.md, TASKS.md, สร้าง repo
- [ ] รีวิวและ merge PR ของรุจิรัตน์
- [ ] ก่อน 12:20: รัน `ensemble.py` + notebook บน `main` ล่าสุด แล้วตรวจว่าเปิด repo แบบสาธารณะได้

## รุจิรัตน์ แสงชัย (6752300739): Boosting + ข้อมูล + วิเคราะห์ผล · branch `rujirat/...`
- [ ] อ่านและทำความเข้าใจ `AdaBoostSAMME` ใน `ensemble.py` (ลองคำนวณ ε, α รอบแรกด้วยมือเทียบกับ `ada.errors_[0]`, `ada.alphas_[0]`)
- [ ] notebook: เพิ่มส่วน "0. สำรวจข้อมูล" ก่อนส่วนที่ 1 (จำนวนต่อ class, scatter petal-length vs petal-width, ข้อสังเกตเรื่อง sepal ที่ถูกปัดเป็นจำนวนเต็ม)
- [ ] notebook ส่วน 5 "วิเคราะห์ผล": ตอบ 5 คำถามจากผลที่รันได้จริง
- [ ] (ถ้ามีเวลา) ลอง AdaBoost กับ `max_depth=2` เทียบกับ stump แล้วสรุปสั้นๆ
- [ ] เปิด PR เข้า `main` **ภายใน 12:00** เพื่อให้มีเวลา merge

## ลำดับเวลา
| เวลา | งาน |
|---|---|
| ตอนนี้ – 12:00 | รุจิรัตน์ทำงานใน branch ของตัวเอง |
| 12:00 | รุจิรัตน์เปิด PR |
| 12:00 – 12:20 | ปฏิมารีวิว + merge + รันทดสอบบน `main` |
| 12:30 | ส่งลิงก์ repo |
