# Ensemble-Learning: กติกาสำหรับ Claude Code ของทุกคนในทีม

Classwork สุดท้ายวิชา Machine Learning: Bagging (parallel) + Boosting (sequential) บน Iris
**กำหนดส่ง 12:30 น. วันที่ 2026-11-13** · อาจารย์ตรวจคะแนนจากคอมมิตสาธารณะบน `main`

## ก่อนเริ่มงาน (Claude ต้องทำทุกครั้ง)
1. ถามผู้ใช้ว่าเป็นใคร แล้วเปิด `TASKS.md` ดูงานของคนนั้น **ทำเฉพาะงานของคนนั้น**
2. `git pull origin main` แล้วสร้าง branch ของตัวเอง: `<ชื่อ>/<งาน>` เช่น `rujirat/analysis`
3. อ่าน `README.md` หัวข้อ 3–4 ก่อนแก้โค้ด

## กติกาโค้ด (ตามโจทย์อาจารย์ ห้ามละเมิด)
- **ห้าม** import `BaggingClassifier`, `AdaBoostClassifier` หรือ ensemble ใดๆ จาก sklearn ตัวอัลกอริทึมต้องเขียนเอง
- base classifier (`DecisionTreeClassifier`, `KNeighborsClassifier`, `SVC`) เรียกจาก sklearn ได้
- Bagging ต้องเป็น parallel (`ProcessPoolExecutor`) ส่วน Boosting ต้องเป็น sequential
- ข้อมูลโหลดผ่าน `ensemble.load_data()` เท่านั้น (Iris จาก Hugging Face `scikit-learn/iris`, split 80/20 seed 42)
- คอมเมนต์ในโค้ดเป็นภาษาไทยสั้นๆ อธิบายเหตุผลทุกส่วนที่เขียนให้ เจ้าของงานต้องอธิบายให้อาจารย์ฟังได้
- ห้ามแต่งตัวเลขผลลัพธ์ ทุกตัวเลขใน README/notebook ต้องมาจากการรันจริง

## ไฟล์และเจ้าของ (กันการชนกัน)
| ไฟล์ | เจ้าของ | หมายเหตุ |
|---|---|---|
| `ensemble.py` ส่วน Bagging + `majority_vote` | ปฏิมา | |
| `ensemble.py` ส่วน `AdaBoostSAMME` + `load_data` | รุจิรัตน์ | แก้เฉพาะ class/ฟังก์ชันของตัวเอง |
| `ensemble_iris.ipynb` | รุจิรัตน์ | ไฟล์ notebook merge ยาก **แก้ได้ทีละคน** |
| `README.md` | ปฏิมา (รวบรวม) | รุจิรัตน์ส่งข้อความวิเคราะห์ผ่าน PR ได้ |
| `CLAUDE.md`, `TASKS.md`, `requirements.txt` | ปฏิมา | |

## วิธีรัน (Windows)
```bash
python -m venv .venv                      # ถ้า uv venv โดน Application Control บล็อก ให้ใช้คำสั่งนี้แทน
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe ensemble.py
.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace ensemble_iris.ipynb
```
- ใช้ `python -m nbconvert` ไม่ใช่ `jupyter nbconvert` (`jupyter-nbconvert.exe` อาจโดนบล็อก)
- ถ้า import sklearn แล้วเจอ `DLL load failed ... Application Control` ให้รันซ้ำอีกครั้ง ส่วน `datasets` โหลดไม่ได้ก็ไม่เป็นไร `load_data()` มี fallback
- ProcessPool บน Windows: ฟังก์ชันที่ส่งเข้า pool ต้องอยู่ใน `ensemble.py` (ไม่ใช่ใน notebook) และสคริปต์ต้องมี `if __name__ == "__main__":`

## ส่งงาน
- **commit ทันทีหลังแก้แต่ละขั้น** (ไฟล์ใหม่ 1 ไฟล์, ฟีเจอร์/แก้บั๊ก 1 เรื่อง, อัปเดต notebook 1 ส่วน) ข้อความบอกว่าขั้นนั้นเปลี่ยนอะไร เช่น `notebook: add EDA section` ห้ามรอทำเสร็จทั้งหมดแล้ว commit ก้อนเดียว จะได้ย้อนหรือแก้เฉพาะขั้นได้
- commit ด้วยชื่อ/อีเมลที่ผูกกับบัญชี GitHub ของตัวเอง (อาจารย์นับคะแนนจากคอมมิต) ตรวจด้วย `git config user.email`
- push branch ของตัวเอง แล้วเปิด PR เข้า `main` (`gh pr create --base main`) ให้ปฏิมาเป็นคน merge
- **ห้าม push เข้า `main` ตรง และห้าม force push**
- ก่อนเปิด PR: รัน `ensemble.py` และ notebook ให้ผ่านทุก cell แล้วอัปเดตสถานะใน `TASKS.md`
- Claude ห้าม commit/push/เปิด PR เอง ถ้าผู้ใช้ไม่ได้สั่ง
