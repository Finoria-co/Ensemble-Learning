# Ensemble Learning บน Iris: Bagging (Parallel) และ Boosting (Sequential)

Classwork สุดท้าย วิชา Machine Learning (รศ.ดร.ปริญญา สงวนสัตย์) · ทีม Finoria

| ชื่อ | รหัสนักศึกษา | ส่วนที่รับผิดชอบ |
|---|---|---|
| ปฏิมา ถิ่นสำราญ | 6752300461 | Bagging แบบ parallel, ดูแล repo / merge |
| รุจิรัตน์ แสงชัย | 6752300739 | Boosting (AdaBoost SAMME), ข้อมูล, วิเคราะห์ผล |

## 1. โจทย์
เขียนโปรแกรม Ensemble Learning 2 แบบ คือ Bagging และ Boosting บนชุดข้อมูล Iris
- base classifier (เช่น KNN, Decision Tree, SVM) เรียกจาก library ได้ → ใช้ `DecisionTreeClassifier` ของ sklearn
- ตัวอัลกอริทึม Bagging และ Boosting **ห้ามใช้ library สำเร็จรูป** → เขียนเองทั้งหมดใน `ensemble.py`
- Bagging ต้องทำงานแบบ **parallel** (NumPy + parallel ที่มากับ Python) → `concurrent.futures.ProcessPoolExecutor`
- Boosting ทำงานแบบ **sequential**

## 2. ข้อมูล
Iris จาก Hugging Face [`scikit-learn/iris`](https://huggingface.co/datasets/scikit-learn/iris): 150 ตัวอย่าง, 4 feature (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm` ไม่ใช้คอลัมน์ `Id`), 3 class (class ละ 50)
- แบ่ง train/test **80/20** แบบ stratified (`random_state=42`) → train 120 (40 ต่อ class), test 30 (10 ต่อ class)
- Bagging และ Boosting เทรนด้วยข้อมูล 80% เท่านั้น test 20% ใช้วัดผลอย่างเดียว
- มีแถวซ้ำ 3 แถว แต่เป็น class เดียวกัน ไม่มีแถวที่ feature เหมือนกันแต่ class ต่างกัน

## 3. ทฤษฎี (เนื้อหาเสริม)
สไลด์บท 03 Decision Tree กล่าวถึง Boosting ใน C5.0 สั้นๆ รายละเอียดด้านล่างเป็นเนื้อหาเสริม

**Bagging (Bootstrap AGGregatING)**: ลด variance
1. สุ่มข้อมูลแบบใส่คืน (bootstrap) ขนาด n จากข้อมูลเทรน n ตัว ทั้งหมด B ชุด
2. เทรนโมเดลย่อย 1 ตัวต่อ 1 ชุด ใช้ Decision Tree เต็มต้น ซึ่ง variance สูง จึงได้ประโยชน์จากการเฉลี่ย
3. ทำนายด้วยการโหวตเสียงข้างมาก

โมเดลแต่ละตัวไม่ขึ้นต่อกัน จึง**เทรนขนานกันได้**

**AdaBoost SAMME (Boosting หลาย class)**: ลด bias ใช้ Decision Stump (ต้นลึก 1 ชั้น) เป็น weak learner
1. เริ่มให้ทุกตัวอย่างมีน้ำหนัก wᵢ = 1/n
2. รอบ t: เทรน stump ด้วย `sample_weight = w`
3. weighted error εₜ = Σ wᵢ·[ทายผิด] / Σ wᵢ (ถ้า εₜ ≥ 1 − 1/K แปลว่าไม่ดีกว่าเดาสุ่ม ให้หยุด)
4. αₜ = ln((1 − εₜ)/εₜ) + ln(K − 1) ซึ่ง K = 3 เทอม ln(K − 1) ทำให้ α > 0 เมื่อดีกว่าเดาสุ่ม (ε < 2/3)
5. wᵢ ← wᵢ·exp(αₜ·[ทายผิด]) แล้ว normalize ตัวที่ทายผิดจึงได้น้ำหนักเพิ่ม

ทำนาย: H(x) = argmaxₖ Σₜ αₜ·[hₜ(x) = k]

รอบ t ต้องใช้น้ำหนักจากรอบ t − 1 จึง**ต้องเทรนตามลำดับ (sequential)**

## 4. โครงสร้างโค้ด
```
├── ensemble.py            # BaggingClassifier (parallel), AdaBoostSAMME (sequential), majority_vote, load_data
├── ensemble_iris.ipynb    # รันทดลอง + กราฟ + confusion matrix + วิเคราะห์ผล
├── requirements.txt
├── CLAUDE.md / TASKS.md   # กติกาทำงานร่วมกันและการแบ่งงานในทีม
```
- `majority_vote(preds, K, weights)`: one-hot ผลทำนาย (B, m, K) → รวมคะแนนถ่วงน้ำหนัก → argmax ใช้ร่วมกันทั้ง Bagging (น้ำหนัก 1) และ AdaBoost (น้ำหนัก α)
- `BaggingClassifier.fit`: สร้าง bootstrap index ทุกชุดในครั้งเดียว `rng.integers(0, n, size=(B, n))` แล้วส่งงานให้ `ProcessPoolExecutor.map`
  - ส่ง X, y ไปเก็บใน worker ครั้งเดียวผ่าน `initializer`
  - `_fit_one` อยู่ระดับ module เพราะบน Windows ต้อง pickle ฟังก์ชันส่งไป process อื่น
- `AdaBoostSAMME.fit`: ลูป `for t in range(T)` ตามขั้นตอนในหัวข้อ 3
- **ใช้ library แค่ไหน**: เฉพาะ base model (`DecisionTreeClassifier`; ส่วน 3.1 ใช้ `SVC`) ตัว Bagging (bootstrap, แจกงานขนาน, โหวต) และ Boosting (น้ำหนัก, ε, α, โหวตถ่วง α) เขียนเองด้วย NumPy ทั้งหมด Bagging สร้างโมเดลใหม่เองจาก class + ค่าตั้ง (`base`, `base_params`) ไม่ใช้ `sklearn.base.clone`
- `verbose=True` ทั้งสองคลาส พิมพ์กลไกทีละขั้น ไว้ดูการทำงานหรือเดโม

## 5. วิธีรัน
```bash
pip install -r requirements.txt
python ensemble.py                    # พิมพ์ตาราง accuracy และเวลาเทรน
python ensemble.py --verbose          # แสดงกลไกทีละขั้น: bootstrap/โหวตของ Bagging และ ε, α, น้ำหนักทีละรอบของ AdaBoost
jupyter notebook ensemble_iris.ipynb  # กราฟและการวิเคราะห์
```
ถ้า import `datasets` ไม่ได้ `load_data()` จะอ่านไฟล์ CSV เดียวกันจาก Hugging Face ตรงๆ แทน

## 6. ผลลัพธ์: ประสิทธิภาพและความเร็ว (`python ensemble.py`)
| Model | Train acc | Test acc | Fit (s) | Predict (s) |
|---|---|---|---|---|
| Decision Tree (single) | 1.0000 | 0.9667 | 0.006 | 0.0005 |
| Bagging 50 trees (parallel) | 1.0000 | 0.9667 | 1.790 | 0.0056 |
| Bagging 50 trees (sequential) | 1.0000 | 0.9667 | 0.034 | 0.0037 |
| Decision Stump (single) | 0.6667 | 0.6667 | 0.001 | 0.0001 |
| AdaBoost SAMME 50 rounds | 1.0000 | 0.9333 | 0.035 | 0.0036 |

- **ประสิทธิภาพ**: Bagging ได้ 0.9667 เท่าต้นเดียวที่ seed 0 แต่เสถียรกว่า เฉลี่ย 10 seed Bagging ได้ 0.9667 ทุก seed ส่วนต้นเดียวเฉลี่ย 0.9533 · AdaBoost ยก stump จาก 0.67 เป็น 0.93
- **ความเร็ว**: Bagging แบบ sequential กับ AdaBoost ใช้เวลา fit ใกล้กัน (~0.035 s) ส่วน Bagging แบบ parallel ใช้ ~1.8 s เพราะต้นทุนการเปิด process คงที่ สูงกว่างานเทรน DT ซึ่งใช้ต้นละไม่ถึง 1 ms (ดูกราฟ B = 10–1000 ใน notebook)
- AdaBoost ทำ parallel ไม่ได้ เพราะรอบ t ต้องรอน้ำหนักจากรอบ t−1
- เวลาต่างไปเล็กน้อยทุกครั้งที่รัน

### Parallel เร็วกว่าเมื่อไหร่ (notebook ส่วน 3.1)
Bagging 16 โมเดล ใช้ base = SVM (`SVC(C=10)`) บน Iris ที่ขยายขนาด (สุ่มแบบใส่คืน + noise σ = 0.8) ใช้ 4 process (เวลา parallel รวมเวลาเปิด process แล้ว · ตัวเลขจากการรัน notebook ล่าสุด เวลาจะต่างไปเล็กน้อยทุกครั้งที่รัน)

| n | parallel | sequential | speedup |
|---|---|---|---|
| 2,000 | 1.85 s | 0.32 s | ×0.17 |
| 5,000 | 5.53 s | 1.91 s | ×0.34 |
| 10,000 | 7.32 s | 8.40 s | **×1.15** |
| 20,000 | 14.92 s | 33.24 s | **×2.23** |

- worker เปิดต่อกันทีละตัว ตัวละราว 1.3 s (ต้อง import numpy/sklearn ใหม่) parallel จึงคุ้มเมื่อโมเดลย่อยแต่ละตัวเทรนนานกว่าต้นทุนนี้มากพอ
- ใช้ SVM แทน DT ในการทดลองนี้ เพราะ SVM เทรนนาน (~n²) แต่โมเดลเล็ก ส่วน DT บนข้อมูลใหญ่โตลึกมาก ต้นทุนไปอยู่ที่การส่งต้นไม้กลับข้าม process
- `BaggingClassifier(base=SVC, base_params={"C": 10})` รับ class ของ base model ตัวไหนก็ได้ ค่าเริ่มต้นคือ `DecisionTreeClassifier`

> หัวข้อวิเคราะห์ผลเพิ่มเติมอยู่ใน notebook ส่วนที่ 5
