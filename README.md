# Ensemble Learning บน Iris: Bagging (Parallel) และ Boosting (Sequential)

Classwork สุดท้าย วิชา Machine Learning (รศ.ดร.ปริญญา สงวนสัตย์) · ทีม Finoria

| ชื่อ | รหัสนักศึกษา | ส่วนที่รับผิดชอบ |
|---|---|---|
| ปฏิมา ถิ่นสำราญ | 6752300461 | Bagging แบบ parallel, ดูแล repo / merge |
| รุจิรัตน์ แสงชัย | 6752300739 | Boosting (AdaBoost SAMME), ข้อมูล, วิเคราะห์ผล |

## 1. โจทย์
เขียนโปรแกรม Ensemble Learning 2 แบบ คือ Bagging และ Boosting บนชุดข้อมูล Iris
- ตัวอัลกอริทึม Bagging และ Boosting **ห้ามใช้ library สำเร็จรูป** → เขียนเองทั้งหมดใน `ensemble.py`
- **งานนี้เขียนเองทั้งหมดด้วย NumPy ไม่ใช้ library ML เลย**: Decision Tree (base model), Bagging, AdaBoost และการแบ่ง train/test (โจทย์อนุญาตให้เรียก base classifier จาก library ได้ แต่เราเขียนเองเพื่อให้เห็นกลไกครบทุกชั้น)
- Bagging ต้องทำงานแบบ **parallel** (NumPy + parallel ที่มากับ Python) → `concurrent.futures.ProcessPoolExecutor`
- Boosting ทำงานแบบ **sequential**

## 2. ข้อมูล
Iris จาก Hugging Face [`scikit-learn/iris`](https://huggingface.co/datasets/scikit-learn/iris): 150 ตัวอย่าง, 4 feature (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm` ไม่ใช้คอลัมน์ `Id`), 3 class (class ละ 50)
- แบ่ง train/test **80/20** แบบ stratified ด้วย `stratified_split()` ที่เขียนเอง (seed 42) → train 120 (40 ต่อ class), test 30 (10 ต่อ class)
- Bagging และ Boosting เทรนด้วยข้อมูล 80% เท่านั้น test 20% ใช้วัดผลอย่างเดียว
- มีแถวซ้ำ 3 แถว แต่เป็น class เดียวกัน ไม่มีแถวที่ feature เหมือนกันแต่ class ต่างกัน

## 3. ทฤษฎี (เนื้อหาเสริม)
สไลด์บท 03 Decision Tree กล่าวถึง Boosting ใน C5.0 สั้นๆ รายละเอียดด้านล่างเป็นเนื้อหาเสริม

**Bagging (Bootstrap AGGregatING)**: ลด variance
1. สุ่มข้อมูลแบบใส่คืน (bootstrap) ขนาด n จากข้อมูลเทรน n ตัว ทั้งหมด B ชุด
2. เทรนโมเดลย่อย 1 ตัวต่อ 1 ชุด ใช้ Decision Tree ต้นเต็ม (`DecisionTree()` ที่เขียนเอง) ซึ่ง variance สูง จึงได้ประโยชน์จากการเฉลี่ย
3. ทำนายด้วยการโหวตเสียงข้างมาก

โมเดลแต่ละตัวไม่ขึ้นต่อกัน จึง**เทรนขนานกันได้**

**AdaBoost SAMME (Boosting หลาย class)**: ลด bias ใช้ Decision Stump (`DecisionTree(max_depth=1)` ที่เขียนเอง) เป็น weak learner
1. เริ่มให้ทุกตัวอย่างมีน้ำหนัก wᵢ = 1/n
2. รอบ t: เทรน stump ด้วย `sample_weight = w`
3. weighted error εₜ = Σ wᵢ·[ทายผิด] / Σ wᵢ (ถ้า εₜ ≥ 1 − 1/K แปลว่าไม่ดีกว่าเดาสุ่ม ให้หยุด)
4. αₜ = ln((1 − εₜ)/εₜ) + ln(K − 1) ซึ่ง K = 3 เทอม ln(K − 1) ทำให้ α > 0 เมื่อดีกว่าเดาสุ่ม (ε < 2/3)
5. wᵢ ← wᵢ·exp(αₜ·[ทายผิด]) แล้ว normalize ตัวที่ทายผิดจึงได้น้ำหนักเพิ่ม

ทำนาย: H(x) = argmaxₖ Σₜ αₜ·[hₜ(x) = k]

รอบ t ต้องใช้น้ำหนักจากรอบ t − 1 จึง**ต้องเทรนตามลำดับ (sequential)**

## 4. โครงสร้างโค้ด
```
├── ensemble.py            # DecisionTree, BaggingClassifier (parallel), AdaBoostSAMME (sequential), majority_vote, stratified_split, load_data
├── ensemble_iris.ipynb    # รันทดลอง + กราฟ + confusion matrix + วิเคราะห์ผล
├── requirements.txt
├── CLAUDE.md / TASKS.md   # กติกาทำงานร่วมกันและการแบ่งงานในทีม
```
- `DecisionTree` (CART): ลองทุก feature และทุกจุดตัดระหว่างค่าที่ต่างกัน เลือกคำถาม `feature <= threshold` ที่ทำให้ **Gini impurity** (Gini = 1 − Σ pₖ², ถ่วงน้ำหนักตัวอย่าง) ลดลงมากที่สุด ใช้ `cumsum` หาจำนวนต่อ class ฝั่งซ้ายของทุกจุดตัดในครั้งเดียว แตกกิ่งแบบ recursive จนบริสุทธิ์หรือถึง `max_depth` รับ `sample_weight` ให้ AdaBoost ได้ (Gini เป็นเกณฑ์เดียวกับ CART/sklearn ส่วนสไลด์บท 03 ใช้ entropy/information gain หลักการเดียวกัน)
- `majority_vote(preds, K, weights)`: one-hot ผลทำนาย (B, m, K) → รวมคะแนนถ่วงน้ำหนัก → argmax ใช้ร่วมกันทั้ง Bagging (น้ำหนัก 1) และ AdaBoost (น้ำหนัก α)
- `BaggingClassifier.fit`: สร้าง bootstrap index ทุกชุดในครั้งเดียว `rng.integers(0, n, size=(B, n))` แล้วส่งงานให้ `ProcessPoolExecutor.map`
  - ส่ง X, y ไปเก็บใน worker ครั้งเดียวผ่าน `initializer`
  - `_fit_one` อยู่ระดับ module เพราะบน Windows ต้อง pickle ฟังก์ชันส่งไป process อื่น
- `AdaBoostSAMME.fit`: ลูป `for t in range(T)` ตามขั้นตอนในหัวข้อ 3
- **ใช้ library แค่ไหน**: NumPy สำหรับคำนวณ, `concurrent.futures` (built-in) สำหรับ parallel, `datasets` สำหรับโหลด Iris และ matplotlib สำหรับกราฟ ไม่มี sklearn ใน `ensemble.py` และ notebook
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
| Decision Tree (single) | 1.0000 | 0.9333 | 0.001 | 0.0000 |
| Bagging 50 trees (parallel) | 1.0000 | 0.9333 | 0.356 | 0.0015 |
| Bagging 50 trees (sequential) | 1.0000 | 0.9333 | 0.039 | 0.0013 |
| Decision Stump (single) | 0.6667 | 0.6667 | 0.000 | 0.0000 |
| AdaBoost SAMME 50 rounds | 1.0000 | 0.9333 | 0.011 | 0.0005 |

- **ประสิทธิภาพ**: ที่ seed 0 ได้ 0.9333 เท่ากันทุกวิธี (ผิด virginica → versicolor 2 ตัว) แต่เฉลี่ย 10 seed Bagging ได้ 0.9433 ส่วนต้นเดียว 0.9333 · AdaBoost ยก stump จาก 0.67 เป็น 0.93 (0.9667 ที่ 10 รอบ)
- **ความเร็ว**: AdaBoost fit เร็วสุดในกลุ่ม ensemble (~0.01 s) เพราะ stump แบ่งครั้งเดียว · Bagging sequential ~0.04 s · Bagging parallel ~0.3 s เพราะต้นทุนเปิด worker 4 process (ไม่ต้อง import sklearn แล้ว จึงต่ำกว่าเดิมที่ ~1.8 s) ที่ B = 1000 parallel เร็วกว่า sequential แล้ว (ดูกราฟ B = 10–1000 ใน notebook)
- AdaBoost ทำ parallel ไม่ได้ เพราะรอบ t ต้องรอน้ำหนักจากรอบ t−1
- เวลาต่างไปเล็กน้อยทุกครั้งที่รัน

### Parallel เร็วกว่าเมื่อไหร่ (notebook ส่วน 3.1)
Bagging 16 ต้น ใช้ base = `DecisionTree` ต้นเต็มที่เขียนเอง บน Iris ที่ขยายขนาด (สุ่มแบบใส่คืน + noise σ = 0.8) ใช้ 4 process (เวลา parallel รวมเวลาเปิด process แล้ว · ตัวเลขจากการรัน notebook ล่าสุด เวลาจะต่างไปเล็กน้อยทุกครั้งที่รัน)

| n | parallel | sequential | speedup |
|---|---|---|---|
| 500 | 0.28 s | 0.13 s | ×0.47 |
| 2,000 | 0.41 s | 0.59 s | **×1.44** |
| 5,000 | 0.84 s | 1.44 s | **×1.71** |
| 10,000 | 1.27 s | 3.04 s | **×2.39** |
| 20,000 | 2.38 s | 6.19 s | **×2.60** |

- ต้นทุนคงที่ของ parallel คือการเปิด worker (เปิด Python ใหม่และ import numpy) ราว 0.3 s parallel จึงคุ้มเมื่องานรวมใหญ่กว่านั้น ในที่นี้ตั้งแต่ n = 2,000
- ไม่ถึง ×4 ทั้งที่ใช้ 4 process เพราะยังมีต้นทุนเปิด process และส่งข้อมูล/ต้นไม้ข้าม process
- `BaggingClassifier(base=..., base_params={...})` รับ class ของ base model ได้ ค่าเริ่มต้นคือ `DecisionTree`

> หัวข้อวิเคราะห์ผลเพิ่มเติมอยู่ใน notebook ส่วนที่ 5
