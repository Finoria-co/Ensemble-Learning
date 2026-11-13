"""
Ensemble Learning บนชุดข้อมูล Iris: Bagging (parallel) และ Boosting (AdaBoost SAMME, sequential)

- Base classifier เรียกจาก sklearn ได้ (DecisionTreeClassifier)
- ตัวอัลกอริทึม Bagging / Boosting เขียนเองทั้งหมดด้วย NumPy
- Bagging เทรนโมเดลย่อยพร้อมกันหลาย process ด้วย concurrent.futures.ProcessPoolExecutor (built-in ของ Python)
- Boosting เทรนทีละรอบตามลำดับ เพราะรอบที่ t ต้องใช้น้ำหนักตัวอย่างจากรอบ t-1

โค้ดวาดกราฟอยู่ใน ensemble_iris.ipynb
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------- helpers
def majority_vote(preds, n_classes, weights=None):
    """
    preds  : (B, m) ผลทำนายของโมเดลย่อย B ตัว บนข้อมูล m ตัวอย่าง
    weights: (B,) น้ำหนักเสียงของแต่ละโมเดล (Bagging = 1 เท่ากันหมด, AdaBoost = alpha)
    คืนค่า : (m,) class ที่ได้คะแนนโหวตรวมมากที่สุด
    """
    if weights is None:
        weights = np.ones(len(preds))
    # one-hot: votes[b, i, k] = 1 ถ้าโมเดล b ทำนายตัวอย่าง i เป็น class k
    votes = (preds[:, :, None] == np.arange(n_classes)).astype(float)
    # รวมคะแนนของทุกโมเดล (ถ่วงน้ำหนัก) -> (m, K) แล้วเลือก class ที่คะแนนสูงสุด
    scores = np.tensordot(weights, votes, axes=1)
    return scores.argmax(axis=1)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------- Bagging
_X = _y = _BASE = None  # ข้อมูลเทรนและ base model ที่แต่ละ worker process เก็บไว้ใช้เอง


def _init_worker(X, y, base):
    """รันครั้งเดียวตอนเปิด worker: ส่งข้อมูลเทรนและ base model ไปเก็บไว้ จะได้ไม่ต้องส่งซ้ำทุกงาน"""
    global _X, _y, _BASE
    _X, _y, _BASE = X, y, base


def _fit_one(args):
    """
    เทรน base model 1 ตัวบน bootstrap sample (งานที่ส่งไปให้แต่ละ process ทำ)
    ต้องอยู่ระดับ module เพราะ ProcessPoolExecutor บน Windows ต้อง pickle ฟังก์ชันส่งไป process อื่น
    """
    idx, seed = args
    model = clone(_BASE)  # สำเนาใหม่ที่ยังไม่ได้เทรน
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    return model.fit(_X[idx], _y[idx])


class BaggingClassifier:
    """
    Bootstrap AGGregatING
    1. สุ่มข้อมูลแบบใส่คืน (bootstrap) ขนาด n จากข้อมูลเทรน n ตัว ทั้งหมด B ชุด
    2. เทรนโมเดลย่อย 1 ตัวต่อ 1 ชุด  <- แต่ละตัวไม่ขึ้นต่อกัน จึงเทรนขนานกันได้
    3. ทำนายด้วยการโหวตเสียงข้างมาก
    """

    def __init__(self, n_estimators=50, max_workers=None, parallel=True, seed=0, base=None):
        self.n_estimators = n_estimators
        # base model: ค่าเริ่มต้นคือ Decision Tree ต้นเต็ม (ไม่จำกัดความลึก) -> variance สูง bagging จึงช่วยได้มาก
        self.base = base if base is not None else DecisionTreeClassifier()
        # จำนวน process: ค่าเริ่มต้นไม่เกิน 4 เพราะการเปิด process แต่ละตัวมีต้นทุน (ต้อง import numpy/sklearn ใหม่)
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.parallel = parallel
        self.seed = seed

    def fit(self, X, y):
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.n_classes_ = int(y.max()) + 1
        # ขั้น 1: สร้าง bootstrap index ของทุกต้นในครั้งเดียวด้วย NumPy -> (B, n)
        self.boot_idx_ = rng.integers(0, n, size=(self.n_estimators, n))
        seeds = rng.integers(0, 2**31 - 1, size=self.n_estimators)
        jobs = [(idx, int(s)) for idx, s in zip(self.boot_idx_, seeds)]

        # ขั้น 2: เทรนทุกต้น
        if self.parallel:
            # แบ่งงานให้หลาย process (หลาย CPU core) ทำพร้อมกัน
            # chunksize: ส่งงานเป็นก้อน ลดจำนวนครั้งที่ต้องสื่อสารข้าม process
            chunk = max(1, self.n_estimators // (4 * self.max_workers))
            with ProcessPoolExecutor(max_workers=self.max_workers,
                                     initializer=_init_worker, initargs=(X, y, self.base)) as ex:
                self.estimators_ = list(ex.map(_fit_one, jobs, chunksize=chunk))
        else:
            # แบบไม่ขนาน ไว้จับเวลาเทียบ
            _init_worker(X, y, self.base)
            self.estimators_ = [_fit_one(job) for job in jobs]
        return self

    def _all_preds(self, X):
        return np.array([est.predict(X) for est in self.estimators_])  # (B, m)

    def predict(self, X):
        # ขั้น 3: โหวตเสียงข้างมาก (ทุกต้นมีเสียงเท่ากัน)
        return majority_vote(self._all_preds(X), self.n_classes_)

    def staged_score(self, X, y):
        """accuracy เมื่อใช้ 1, 2, ..., B ต้นแรก (ไว้พล็อตกราฟ)"""
        preds = self._all_preds(X)
        return np.array([accuracy(y, majority_vote(preds[:b], self.n_classes_))
                         for b in range(1, len(preds) + 1)])


# ---------------------------------------------------------------- Boosting
class AdaBoostSAMME:
    """
    AdaBoost แบบหลาย class (SAMME: Zhu et al., 2009) ใช้ Decision Stump (ต้นลึก 1 ชั้น) เป็น weak learner
    เทรนตามลำดับ: ตัวอย่างที่รอบก่อนทายผิดจะได้น้ำหนักเพิ่ม รอบถัดไปจึงโฟกัสตัวที่ยาก
    """

    def __init__(self, n_estimators=50, max_depth=1, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed

    def fit(self, X, y):
        n = len(X)
        K = int(y.max()) + 1
        self.n_classes_ = K
        w = np.full(n, 1.0 / n)  # ขั้น 1: เริ่มให้ทุกตัวอย่างมีน้ำหนักเท่ากัน
        self.estimators_, self.alphas_, self.errors_ = [], [], []

        for t in range(self.n_estimators):  # sequential: รอบ t ใช้ w จากรอบ t-1
            # ขั้น 2: เทรน weak learner โดยถ่วงน้ำหนักตัวอย่างด้วย w
            stump = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.seed + t)
            stump.fit(X, y, sample_weight=w)
            miss = stump.predict(X) != y  # ตัวไหนทายผิด

            # ขั้น 3: weighted error
            err = np.sum(w[miss]) / np.sum(w)
            if err >= 1 - 1 / K:
                # แย่กว่าหรือเท่ากับการเดาสุ่ม -> ไม่มีประโยชน์ หยุด
                break
            err = max(err, 1e-10)  # กัน log(0) กรณีทายถูกหมด

            # ขั้น 4: น้ำหนักเสียงของโมเดลนี้ (เทอม ln(K-1) ทำให้ alpha > 0 เมื่อดีกว่าเดาสุ่ม 1/K)
            alpha = np.log((1 - err) / err) + np.log(K - 1)
            self.estimators_.append(stump)
            self.alphas_.append(alpha)
            self.errors_.append(err)

            if err <= 1e-10:
                break  # ทายถูกหมดแล้ว ไม่ต้องปรับน้ำหนักต่อ

            # ขั้น 5: เพิ่มน้ำหนักตัวที่ทายผิด แล้ว normalize ให้ผลรวม = 1
            w = w * np.exp(alpha * miss)
            w = w / w.sum()

        self.alphas_ = np.array(self.alphas_)
        self.errors_ = np.array(self.errors_)
        return self

    def _all_preds(self, X):
        return np.array([est.predict(X) for est in self.estimators_])  # (T, m)

    def predict(self, X):
        # H(x) = argmax_k  Σ_t alpha_t · [h_t(x) = k]
        return majority_vote(self._all_preds(X), self.n_classes_, self.alphas_)

    def staged_score(self, X, y):
        preds = self._all_preds(X)
        return np.array([accuracy(y, majority_vote(preds[:t], self.n_classes_, self.alphas_[:t]))
                         for t in range(1, len(preds) + 1)])


# ---------------------------------------------------------------- data
HF_DATASET = "scikit-learn/iris"
HF_CSV_URL = "https://huggingface.co/datasets/scikit-learn/iris/resolve/main/Iris.csv"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]  # ไม่ใช้คอลัมน์ Id


def load_iris_hf():
    """
    โหลด Iris จาก Hugging Face (scikit-learn/iris) -> X (150, 4), species (150,)
    ใช้ datasets.load_dataset ก่อน ถ้าเครื่องไหน import ไม่ได้ ให้อ่านไฟล์ CSV เดียวกันจาก Hugging Face ตรงๆ
    """
    try:
        from datasets import load_dataset
        d = load_dataset(HF_DATASET)["train"]
        X = np.column_stack([np.asarray(d[c], dtype=float) for c in FEATURES])
        species = np.asarray(d["Species"])
    except ImportError:
        import io
        import urllib.request
        text = urllib.request.urlopen(HF_CSV_URL, timeout=30).read().decode("utf-8")
        raw = np.genfromtxt(io.StringIO(text), delimiter=",", names=True, dtype=None, encoding="utf-8")
        X = np.column_stack([raw[c].astype(float) for c in FEATURES])
        species = raw["Species"].astype(str)
    return X, species


def load_data(test_size=0.2, seed=42):
    """Iris แบ่ง train/test 80/20 แบบ stratified (สัดส่วน class เท่ากันทั้งสองชุด) เทรนด้วย 80% เท่านั้น"""
    from sklearn.model_selection import train_test_split

    X, species = load_iris_hf()
    class_names, y = np.unique(species, return_inverse=True)  # ชื่อ species -> 0, 1, 2
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return X_tr, X_te, y_tr, y_te, class_names


# ---------------------------------------------------------------- evaluation
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    """เทรน + ทำนาย แล้วคืน accuracy และเวลา (วินาที) ของแต่ละขั้น"""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_te)
    t_pred = time.perf_counter() - t0
    return {"model": name, "train_acc": accuracy(y_tr, model.predict(X_tr)), "test_acc": accuracy(y_te, pred),
            "fit_s": t_fit, "predict_s": t_pred}


def compare(X_tr, y_tr, X_te, y_te, n_estimators=50):
    """ตารางเปรียบเทียบประสิทธิภาพการทำนายและความเร็วของแต่ละวิธี"""
    return [
        evaluate("Decision Tree (single)", DecisionTreeClassifier(random_state=0), X_tr, y_tr, X_te, y_te),
        evaluate(f"Bagging {n_estimators} trees (parallel)", BaggingClassifier(n_estimators, parallel=True, seed=0),
                 X_tr, y_tr, X_te, y_te),
        evaluate(f"Bagging {n_estimators} trees (sequential)", BaggingClassifier(n_estimators, parallel=False, seed=0),
                 X_tr, y_tr, X_te, y_te),
        evaluate("Decision Stump (single)", DecisionTreeClassifier(max_depth=1, random_state=0), X_tr, y_tr, X_te, y_te),
        evaluate(f"AdaBoost SAMME {n_estimators} rounds", AdaBoostSAMME(n_estimators, seed=0), X_tr, y_tr, X_te, y_te),
    ]


# ---------------------------------------------------------------- main
if __name__ == "__main__":  # จำเป็นบน Windows: process ลูกจะ import ไฟล์นี้ซ้ำ ต้องไม่รันส่วนนี้ซ้ำ
    X_tr, X_te, y_tr, y_te, _ = load_data()
    print(f"Iris ({HF_DATASET}): train {len(X_tr)} / test {len(X_te)} (80/20), CPU cores = {os.cpu_count()}\n")
    print(f"{'Model':<34}{'Train acc':>10}{'Test acc':>10}{'Fit (s)':>10}{'Predict (s)':>13}")
    for r in compare(X_tr, y_tr, X_te, y_te):
        print(f"{r['model']:<34}{r['train_acc']:>10.4f}{r['test_acc']:>10.4f}{r['fit_s']:>10.3f}{r['predict_s']:>13.4f}")
