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
from concurrent.futures.process import BrokenProcessPool

import numpy as np
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


# ---------------------------------------------------------------- Decision Tree (base model, เขียนเอง)
class DecisionTree:
    """
    Decision Tree แบบ CART เขียนเองด้วย NumPy (base model ของทั้ง Bagging และ AdaBoost)
    - แบ่งด้วยคำถาม "feature <= threshold" ที่ทำให้ Gini impurity (ถ่วงน้ำหนักตัวอย่าง) ลดลงมากที่สุด
    - max_depth=None = แตกกิ่งจนทุก leaf บริสุทธิ์ (ต้นเต็ม ใช้ใน Bagging), max_depth=1 = stump (ใช้ใน AdaBoost)
    - รับ sample_weight ได้ AdaBoost จึงส่งน้ำหนักตัวอย่างเข้ามาได้
    - random_state ใช้สลับลำดับการลอง feature (ตัดสินกรณีคำถามสองข้อดีเท่ากัน)

    โครงสร้างต้นไม้เก็บเป็น array ต่อ node: feature_[i], threshold_[i], left_[i], right_[i], value_[i] (น้ำหนักต่อ class)
    feature_[i] = -1 แปลว่า node i เป็น leaf · node 0 คือราก
    """

    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    @staticmethod
    def gini(counts):
        """Gini = 1 - Σ p_k²  (counts: น้ำหนักต่อ class, ใส่ได้หลายแถวพร้อมกัน)"""
        total = counts.sum(axis=-1, keepdims=True)
        p = counts / np.where(total > 0, total, 1)
        return 1.0 - (p ** 2).sum(axis=-1)

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.n_classes_ = int(y.max()) + 1
        self._rng = np.random.default_rng(self.random_state)
        self.feature_, self.threshold_, self.left_, self.right_, self.value_ = [], [], [], [], []
        self._grow(X, y, w, depth=0)
        # list -> array เพื่อทำนายแบบ vectorize
        self.feature_ = np.array(self.feature_, dtype=int)
        self.threshold_ = np.array(self.threshold_, dtype=float)
        self.left_ = np.array(self.left_, dtype=int)
        self.right_ = np.array(self.right_, dtype=int)
        self.value_ = np.array(self.value_, dtype=float)
        return self

    def _grow(self, X, y, w, depth):
        """สร้าง node แล้วแตกกิ่งซ้าย/ขวาแบบ recursive คืนเลข node"""
        counts = np.bincount(y, weights=w, minlength=self.n_classes_)
        node = len(self.feature_)
        self.feature_.append(-1)
        self.threshold_.append(np.nan)
        self.left_.append(-1)
        self.right_.append(-1)
        self.value_.append(counts)

        # หยุดเมื่อ: ถึงความลึกที่กำหนด, ตัวอย่างน้อยเกินไป หรือ node บริสุทธิ์แล้ว (มี class เดียว)
        if ((self.max_depth is not None and depth >= self.max_depth)
                or len(y) < self.min_samples_split or np.count_nonzero(counts) <= 1):
            return node
        split = self._best_split(X, y, w, counts)
        if split is None:  # ไม่มีคำถามไหนลด impurity ได้ -> เป็น leaf
            return node
        f, thr = split
        go_left = X[:, f] <= thr
        self.feature_[node], self.threshold_[node] = f, thr
        self.left_[node] = self._grow(X[go_left], y[go_left], w[go_left], depth + 1)
        self.right_[node] = self._grow(X[~go_left], y[~go_left], w[~go_left], depth + 1)
        return node

    def _best_split(self, X, y, w, counts):
        """
        ลองทุก feature และทุกจุดตัดระหว่างค่าที่ต่างกัน เลือกจุดที่ Gini หลังแบ่ง (ถ่วงตามน้ำหนักแต่ละฝั่ง) ต่ำสุด
        ใช้ cumsum: เรียงค่าแล้วสะสมน้ำหนักต่อ class -> ได้ count ฝั่งซ้ายของทุกจุดตัดในครั้งเดียว
        """
        n, d = X.shape
        total = counts.sum()
        parent = self.gini(counts)
        onehot_w = np.zeros((n, self.n_classes_))
        onehot_w[np.arange(n), y] = w  # น้ำหนักของแต่ละตัวอย่าง วางไว้ในคอลัมน์ class ของมัน
        best_gain, best = 1e-12, None
        for f in self._rng.permutation(d):
            order = np.argsort(X[:, f], kind="stable")
            xs = X[order, f]
            cut = xs[1:] > xs[:-1]  # ตัดได้เฉพาะระหว่างค่าที่ต่างกัน
            if not cut.any():
                continue
            left = np.cumsum(onehot_w[order], axis=0)[:-1][cut]  # น้ำหนักต่อ class ฝั่งซ้าย ของทุกจุดตัด
            right = counts - left
            wl, wr = left.sum(axis=1), right.sum(axis=1)
            impurity = (wl * self.gini(left) + wr * self.gini(right)) / total
            gain = parent - impurity
            gain[(wl <= 0) | (wr <= 0)] = -np.inf
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain = gain[i]
                best = (int(f), float((xs[:-1][cut][i] + xs[1:][cut][i]) / 2))  # threshold = กึ่งกลางระหว่างสองค่า
        return best

    def predict(self, X):
        """ส่งทุกตัวอย่างลงต้นไม้พร้อมกัน จนถึง leaf แล้วตอบ class ที่น้ำหนักมากที่สุดใน leaf นั้น"""
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        while True:
            f = self.feature_[node]
            active = np.flatnonzero(f >= 0)  # ตัวอย่างที่ยังไม่ถึง leaf
            if len(active) == 0:
                break
            n_act = node[active]
            go_left = X[active, f[active]] <= self.threshold_[n_act]
            node[active] = np.where(go_left, self.left_[n_act], self.right_[n_act])
        return self.value_[node].argmax(axis=1)


# ---------------------------------------------------------------- Bagging
_X = _y = _BASE_CLS = _BASE_PARAMS = None  # ข้อมูลเทรนและชนิด/ค่าตั้งของ base model ที่แต่ละ worker process เก็บไว้ใช้เอง


def _init_worker(X, y, base_cls, base_params):
    """รันครั้งเดียวตอนเปิด worker: ส่งข้อมูลเทรนและชนิดของ base model ไปเก็บไว้ จะได้ไม่ต้องส่งซ้ำทุกงาน"""
    global _X, _y, _BASE_CLS, _BASE_PARAMS
    _X, _y, _BASE_CLS, _BASE_PARAMS = X, y, base_cls, base_params


def _fit_one(args):
    """
    เทรน base model 1 ตัวบน bootstrap sample (งานที่ส่งไปให้แต่ละ process ทำ)
    ต้องอยู่ระดับ module เพราะ ProcessPoolExecutor บน Windows ต้อง pickle ฟังก์ชันส่งไป process อื่น
    """
    idx, seed = args
    model = _BASE_CLS(**_BASE_PARAMS)  # สร้างโมเดลใหม่ที่ยังไม่ได้เทรน (เช่น DecisionTreeClassifier())
    if hasattr(model, "random_state"):
        model.random_state = seed  # แต่ละต้นใช้ seed ของตัวเอง
    return model.fit(_X[idx], _y[idx])  # เทรนบน bootstrap ของตัวเองเท่านั้น


class BaggingClassifier:
    """
    Bootstrap AGGregatING
    1. สุ่มข้อมูลแบบใส่คืน (bootstrap) ขนาด n จากข้อมูลเทรน n ตัว ทั้งหมด B ชุด
    2. เทรนโมเดลย่อย 1 ตัวต่อ 1 ชุด  <- แต่ละตัวไม่ขึ้นต่อกัน จึงเทรนขนานกันได้
    3. ทำนายด้วยการโหวตเสียงข้างมาก
    """

    def __init__(self, n_estimators=50, max_workers=None, parallel=True, seed=0,
                 base=DecisionTreeClassifier, base_params=None, verbose=False):
        self.n_estimators = n_estimators
        self.verbose = verbose  # True = พิมพ์ขั้นตอนการทำงาน (ไว้ดูกลไก/เดโม)
        # base model = class ของ library (ใช้ได้ตามโจทย์) + ค่าตั้ง เช่น base=SVC, base_params={"C": 10}
        # ค่าเริ่มต้นคือ Decision Tree ต้นเต็ม (ไม่จำกัดความลึก) -> variance สูง bagging จึงช่วยได้มาก
        self.base = base
        self.base_params = dict(base_params or {})
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
        if self.verbose:
            uniq = np.array([len(np.unique(r)) for r in self.boot_idx_])
            print(f"[Bagging] ขั้น 1 bootstrap: สุ่มเลขแถวแบบใส่คืน {self.n_estimators} ชุด ชุดละ {n} แถว -> boot_idx_ {self.boot_idx_.shape}")
            for b in range(min(3, self.n_estimators)):
                print(f"  ชุด {b + 1}: แถวแรกๆ {self.boot_idx_[b, :8].tolist()} ... "
                      f"ไม่ซ้ำ {uniq[b]}/{n} ({uniq[b] / n:.0%}), ไม่ถูกเลือก (out-of-bag) {n - uniq[b]}")
            print(f"  เฉลี่ยทุกชุด: แถวไม่ซ้ำ {uniq.mean() / n:.1%} (ทฤษฎี 1 - 1/e = 63.2%)")
            t0 = time.perf_counter()

        # ขั้น 2: เทรนทุกต้น
        self.used_parallel_ = False
        if self.parallel:
            # แบ่งงานให้หลาย process (หลาย CPU core) ทำพร้อมกัน
            # chunksize: ส่งงานเป็นก้อน ลดจำนวนครั้งที่ต้องสื่อสารข้าม process
            chunk = max(1, self.n_estimators // (4 * self.max_workers))
            for attempt in (1, 2):
                try:
                    with ProcessPoolExecutor(max_workers=self.max_workers, initializer=_init_worker,
                                             initargs=(X, y, self.base, self.base_params)) as ex:
                        self.estimators_ = list(ex.map(_fit_one, jobs, chunksize=chunk))
                    self.used_parallel_ = True
                    break
                except BrokenProcessPool as e:
                    # worker ตายกลางทาง (เช่น Windows Application Control บล็อก DLL ตอน worker import ครั้งแรก)
                    # ครั้งแรกลองเปิด pool ใหม่ ครั้งที่สองถอยไปเทรนแบบ sequential และแจ้งให้รู้ว่าไม่ได้ขนาน
                    print(f"[Bagging] process pool ล้ม (ครั้งที่ {attempt}): {e}")
            if not self.used_parallel_:
                print("[Bagging] ใช้ parallel ไม่ได้ในสภาพแวดล้อมนี้ -> เทรนแบบ sequential แทน (เวลาที่วัดได้ไม่ใช่ของ parallel)")
                _init_worker(X, y, self.base, self.base_params)
                self.estimators_ = [_fit_one(job) for job in jobs]
        else:
            # แบบไม่ขนาน ไว้จับเวลาเทียบ
            _init_worker(X, y, self.base, self.base_params)
            self.estimators_ = [_fit_one(job) for job in jobs]
        if self.verbose:
            how = f"parallel {self.max_workers} process" if self.used_parallel_ else "sequential ทีละต้น"
            print(f"[Bagging] ขั้น 2 เทรน {self.base.__name__} {self.n_estimators} ตัว ({how}) "
                  f"ใช้เวลา {time.perf_counter() - t0:.3f} s · แต่ละตัวไม่ต้องรอกัน")
        return self

    def _all_preds(self, X):
        return np.array([est.predict(X) for est in self.estimators_])  # (B, m)

    def predict(self, X, verbose=False):
        # ขั้น 3: โหวตเสียงข้างมาก (ทุกต้นมีเสียงเท่ากัน)
        preds = self._all_preds(X)
        if verbose:
            print(f"[Bagging] ขั้น 3 โหวต: แต่ละตัวอย่างนับเสียงจาก {len(preds)} ต้น (เสียงเท่ากัน)")
            for i in range(min(3, preds.shape[1])):
                votes = np.bincount(preds[:, i], minlength=self.n_classes_)
                print(f"  ตัวอย่าง {i}: เสียงต่อ class {votes.tolist()} -> ทาย class {votes.argmax()}")
        return majority_vote(preds, self.n_classes_)

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

    def __init__(self, n_estimators=50, max_depth=1, seed=0, verbose=False, feature_names=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.verbose = verbose  # True = พิมพ์ขั้นตอนทีละรอบ (ไว้ดูกลไก/เดโม)
        self.feature_names = feature_names  # ชื่อ feature ไว้แสดงคำถามของ stump

    def fit(self, X, y):
        n = len(X)
        K = int(y.max()) + 1
        self.n_classes_ = K
        w = np.full(n, 1.0 / n)  # ขั้น 1: เริ่มให้ทุกตัวอย่างมีน้ำหนักเท่ากัน
        self.estimators_, self.alphas_, self.errors_ = [], [], []
        if self.verbose:
            print(f"[AdaBoost] ขั้น 1 น้ำหนักเริ่มต้น w = 1/{n} = {1 / n:.4f} ทุกตัว · เทรนทีละรอบ (รอบ t ใช้ w จากรอบ t-1)")
            print(f"  {'รอบ':>4}  {'คำถามของ stump':<28}{'ผิด':>5}{'ε':>8}{'α':>8}{'น้ำหนักรวมของตัวที่ผิด (ก่อน→หลัง)':>36}")

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
            before = w[miss].sum() / w.sum()
            w = w * np.exp(alpha * miss)
            w = w / w.sum()
            if self.verbose and (t < 5 or (t + 1) % 10 == 0):
                print(f"  {t + 1:>4}  {self._rule(stump):<28}{int(miss.sum()):>5}{err:>8.3f}{alpha:>8.3f}"
                      f"{before:>24.3f} → {w[miss].sum():.3f}")

        self.alphas_ = np.array(self.alphas_)
        self.errors_ = np.array(self.errors_)
        return self

    def _rule(self, stump):
        """คำถามที่ stump ใช้แบ่ง (ราก) เช่น 'PetalWidthCm <= 0.80'"""
        f, thr = stump.tree_.feature[0], stump.tree_.threshold[0]
        if f < 0:
            return "(ไม่แบ่ง)"
        name = self.feature_names[f] if self.feature_names is not None else f"x[{f}]"
        return f"{name} <= {thr:.2f}" + (" (+ชั้นลึกกว่า)" if self.max_depth > 1 else "")

    def _all_preds(self, X):
        return np.array([est.predict(X) for est in self.estimators_])  # (T, m)

    def predict(self, X, verbose=False):
        # H(x) = argmax_k  Σ_t alpha_t · [h_t(x) = k]
        preds = self._all_preds(X)
        if verbose:
            print(f"[AdaBoost] โหวตถ่วง α: คะแนน class k = ผลรวม α ของ stump ที่ทาย k ({len(preds)} stumps)")
            for i in range(min(3, preds.shape[1])):
                score = np.array([self.alphas_[preds[:, i] == k].sum() for k in range(self.n_classes_)])
                print(f"  ตัวอย่าง {i}: คะแนนต่อ class {np.round(score, 2).tolist()} -> ทาย class {score.argmax()}")
        return majority_vote(preds, self.n_classes_, self.alphas_)

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
    if getattr(model, "parallel", False) and not getattr(model, "used_parallel_", True):
        name += " [ถอยเป็น sequential]"  # pool ล้ม ตัวเลขเวลาแถวนี้จึงไม่ใช่ของ parallel
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
    import sys

    X_tr, X_te, y_tr, y_te, _ = load_data()
    print(f"Iris ({HF_DATASET}): train {len(X_tr)} / test {len(X_te)} (80/20), CPU cores = {os.cpu_count()}\n")

    if "--verbose" in sys.argv:  # python ensemble.py --verbose : แสดงกลไกทีละขั้นของทั้งสองโมเดล
        print("=" * 20, "Bagging (parallel)", "=" * 20)
        bag = BaggingClassifier(50, parallel=True, seed=0, verbose=True).fit(X_tr, y_tr)
        print(f"  test accuracy = {accuracy(y_te, bag.predict(X_te, verbose=True)):.4f}\n")
        print("=" * 20, "Boosting: AdaBoost SAMME (sequential)", "=" * 20)
        ada = AdaBoostSAMME(50, seed=0, verbose=True, feature_names=FEATURES).fit(X_tr, y_tr)
        print(f"  test accuracy = {accuracy(y_te, ada.predict(X_te, verbose=True)):.4f}\n")
        print("=" * 20, "เปรียบเทียบทุกวิธี", "=" * 20)

    print(f"{'Model':<34}{'Train acc':>10}{'Test acc':>10}{'Fit (s)':>10}{'Predict (s)':>13}")
    for r in compare(X_tr, y_tr, X_te, y_te):
        print(f"{r['model']:<34}{r['train_acc']:>10.4f}{r['test_acc']:>10.4f}{r['fit_s']:>10.3f}{r['predict_s']:>13.4f}")
