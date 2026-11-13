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
Iris จาก Hugging Face [`antoniopelusi/iris`](https://huggingface.co/datasets/antoniopelusi/iris): 150 ตัวอย่าง, 4 feature, 3 class (class ละ 50)
- แบ่ง train/test 70/30 แบบ stratified (`random_state=42`) → train 105, test 45
- ข้อสังเกต: ชุดนี้ปัด `sepal-length` และ `sepal-width` เป็นจำนวนเต็ม (Iris มาตรฐานมีทศนิยม 1 ตำแหน่ง) ทำให้มีบางตัวอย่างที่ feature เหมือนกันแต่ class ต่างกัน train accuracy จึงได้สูงสุด 0.9905 ไม่ใช่ 1.0

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

## 5. วิธีรัน
```bash
pip install -r requirements.txt
python ensemble.py                    # พิมพ์ตาราง accuracy และเวลาเทรน
jupyter notebook ensemble_iris.ipynb  # กราฟและการวิเคราะห์
```
ถ้า import `datasets` ไม่ได้ `load_data()` จะอ่านไฟล์ CSV เดียวกันจาก Hugging Face ตรงๆ แทน

## 6. ผลลัพธ์ (`python ensemble.py`)
| Model | Train acc | Test acc |
|---|---|---|
| Decision Tree (single) | 0.9905 | 0.9333 |
| Bagging (50 trees, parallel) | 0.9905 | 0.9111 |
| Decision Stump (single) | 0.6667 | 0.6667 |
| AdaBoost SAMME (50 stumps) | 0.9905 | 0.9111 |

- AdaBoost ยก accuracy ของ stump จาก 0.67 เป็น 0.91
- เวลา fit ของ Bagging 50 ต้น: parallel 1.86 s, sequential 0.04 s
  - บน Iris แบบ parallel ช้ากว่า เพราะต้นทุนการเปิด process (ราว 1.9 s คงที่) สูงกว่างานเทรน DT ซึ่งใช้ต้นละไม่ถึง 1 ms
  - ดูกราฟ B = 10–1000 ใน notebook

> หัวข้อวิเคราะห์ผลเพิ่มเติมอยู่ใน notebook ส่วนที่ 5
