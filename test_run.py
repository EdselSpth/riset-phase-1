import logging
import os
import json
import joblib
import warnings

try:
    from IPython.display import display
except ImportError:
    display = print

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score, average_precision_score  # <-- Tambah average_precision_score di sini
from sklearn.model_selection import GroupShuffleSplit

# Setup MLflow untuk Word2Vec
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("https://mlflow.smbgarasibmw.my.id/")
mlflow.set_experiment("Phase_1_w2v_Text_Representation")  # <-- Ganti nama eksperimen ke w2v

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
DATASET_PATH = "dataset/Dataset_EDA_Word2Vec.csv"  
OUTPUT_DIR = "outputs/phase1_word2vec"
RANDOM_STATE = 42

TRAIN_GROUP_RATIO = 0.8 

os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory initialized at: {OUTPUT_DIR}")
# ==========================================
# STEP 1 — Load Dataset & Validation
# ==========================================
logger.info(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)
logger.info(f"Dataset shape: {df.shape}")

# 1. Detect requirement text column otomatis
text_keywords = ["requirement", "text", "description", "sentence", "story", "content"]
text_cols = [c for c in df.columns if any(k in c.lower() for k in text_keywords)]
if text_cols:
    logger.info(f"Detected requirement text column(s): {text_cols}")
    nama_kolom_teks = text_cols[0]  # Kunci nama kolom untuk dipanggil di sel berikutnya
else:
    raise ValueError("No text column found for Word2Vec pipeline!") # <-- SESUAIKAN TEKS

# 2. Validate required columns bisnis & ranking
required_cols = {"id", "project_id", "type", "value", "effort", "risk", "stakeholder_priority", "rank"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

logger.info("All required columns present for Word2Vec ranking pipeline.") # <-- SESUAIKAN TEKS

# 3. CRITICAL SORTING: Wajib diurutkan berdasarkan project_id (sebagai query group)
df = df.sort_values(by="project_id").reset_index(drop=True)
logger.info("Dataset sorted by 'project_id' for ranking context.")

display(df.head(3))
# ==========================================
# STEP 2 — Data Validation
# ==========================================
validation_report = {}

# Missing values
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]
validation_report["missing_values"] = len(missing_cols)
if len(missing_cols) > 0:
    logger.warning(f"Columns with missing values:\n{missing_cols}")
else:
    logger.info("No missing values found.")

# Duplicate rows
dup_rows = df.duplicated().sum()
validation_report["duplicate_rows"] = dup_rows
if dup_rows > 0:
    logger.warning(f"Duplicate rows: {dup_rows}")
else:
    logger.info("No duplicate rows found.")

# Duplicate requirements (by id)
dup_ids = df["id"].duplicated().sum()
validation_report["duplicate_ids"] = dup_ids
if dup_ids > 0:
    logger.warning(f"Duplicate requirement IDs: {dup_ids}")
else:
    logger.info("No duplicate IDs found.")

# Invalid project_id
invalid_pid = df["project_id"].isnull().sum() + (df["project_id"].astype(str).str.strip() == "").sum()
validation_report["invalid_project_ids"] = int(invalid_pid)
if invalid_pid > 0:
    logger.warning(f"Invalid project IDs: {invalid_pid}")
else:
    logger.info("All project IDs are valid.")

# Invalid rank (should be numeric, non-negative)
invalid_rank = (~pd.to_numeric(df["rank"], errors="coerce").notna()).sum()
validation_report["invalid_rank"] = int(invalid_rank)
if invalid_rank > 0:
    logger.warning(f"Invalid rank values: {invalid_rank}")
else:
    logger.info("All rank values are valid.")

# Data types
validation_report["dtypes"] = {c: str(dt) for c, dt in df.dtypes.items()}

logger.info("=== Validation Report ===")
for k, v in validation_report.items():
    logger.info(f"  {k}: {v}")
logger.info("Preparing pre-computed Word2Vec dataset...")

# Filter semua kolom yang namanya diawali dengan 'w2v_'
w2v_cols = [c for c in df.columns if c.startswith("w2v_")]
X_w2v = df[w2v_cols].values  # Langsung jadi matriks fitur teks Word2Vec

logger.info(f"Loaded pre-computed Word2Vec matrix. Shape: {X_w2v.shape}")

# ==========================================
# STEP 4 — Feature Fusion (Word2Vec Version)
# ==========================================
num_features = ["value", "effort", "risk", "stakeholder_priority"]
logger.info(f"Numerical business features to fuse: {num_features}")

# 1. Ambil nilai numerik dari DataFrame menjadi NumPy Array
X_num = df[num_features].values

# 2. Fitur teks Word2Vec langsung diambil dari STEP 3 (Sudah berbentuk Dense Array)
X_text_dense = X_w2v 

# 3. Proses One-Hot Encoding langsung di sini (Biar mandiri & gak ketergantungan sel lain)
logger.info("Performing One-Hot Encoding on 'type' column...")
encoded_type_df = pd.get_dummies(df["type"], prefix="type")
X_encoded = encoded_type_df.values
logger.info(f"One-hot encoded type columns: {list(encoded_type_df.columns)}")

# 4. FUSI DATA: Gabungkan fitur bisnis, embedding Word2Vec, dan category encoding secara horizontal
X_fused = np.hstack((X_num, X_text_dense, X_encoded))

# 5. Catat dimensi matriks final ke MLflow
mlflow.log_param("total_fused_features", X_fused.shape[1])

logger.info(f"Fusion completed:")
logger.info(f" - Numerical dimension: {X_num.shape[1]}")
logger.info(f" - Word2Vec text dimension: {X_text_dense.shape[1]}")
logger.info(f" - Encoded categorical dimension: {X_encoded.shape[1]}")
logger.info(f" - Final Fused Feature Matrix shape (X): {X_fused.shape}")

# Mengintip beberapa baris pertama hasil fusi (fitur numerik bisnis berada di 4 kolom terdepan)
print("\nFirst 3 rows of final feature matrix (X) looks like:")
print(X_fused[:3, :8]) # Intip 4 fitur numerik + 4 dimensi W2V pertama
logger.info("Transforming ranks into LambdaRank-compatible labels...")

# 1. Transformasi rank ke skor relevansi (0 = terburuk, N-1 = terbaik)
df["label"] = df.groupby("project_id")["rank"].transform(
    lambda x: x.rank(method="dense", ascending=False).astype(int) - 1
)

# 2. Definisikan nilai target y akhir dari kolom label baru
y = df["label"].values
query_ids = df["project_id"]

# 3. Hitung distribusi ukuran tiap kelompok proyek (Query Group)
group_sizes = query_ids.value_counts()
num_groups = len(group_sizes)

logger.info(f"Number of query groups (projects): {num_groups}")
logger.info(f"Group size stats:\n{group_sizes.describe()}")

# Peringatan untuk proyek yang isinya cuma 1 kebutuhan
single_req_groups = (group_sizes == 1).sum()
if single_req_groups > 0:
    logger.warning(f"Groups with only 1 requirement: {single_req_groups} — these cannot be ranked")

# 4. Kunci ukuran grup untuk kebutuhan parameter training LightGBM Ranker
group_counts = group_sizes.sort_index().values  # Urut berdasarkan project_id
logger.info(f"Group sizes array for LightGBM (first 10): {group_counts[:10]}...")

# 5. Log metrik distribusi grup dan label ke MLflow
mlflow.log_param("num_query_groups", num_groups)
mlflow.log_param("label_min_value", int(y.min()))
mlflow.log_param("label_max_value", int(y.max()))

logger.info(f"Label validation -> Range: [{y.min()}, {y.max()}], Unique labels: {sorted(np.unique(y))[:20]}...")
# ==========================================
# STEP 8 — Group-Aware Train-Test Split
# ==========================================
# Definisikan ulang variabel config yang hilang akibat restart kernel
TEST_SIZE = 0.2      # Sesuaikan dengan ratio lu (misal 0.2 atau 0.3)
RANDOM_STATE = 42    # Sesuaikan dengan random state awal lu

logger.info("Performing group-aware train-test split using GroupShuffleSplit...")

# 1. Definisikan array groups berdasarkan project_id
groups = df["project_id"].values

# 2. Inisialisasi GroupShuffleSplit memakai parameter dari global config
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_fused, y, groups=groups))

# 3. Slice matriks X_fused (Wajib slice langsung, JANGAN pakai .iloc karena ini NumPy Array!)
X_train = X_fused[train_idx]
X_test = X_fused[test_idx]

# 4. Slice target label dan kelompok data
y_train = y[train_idx]
y_test = y[test_idx]
groups_train = groups[train_idx]
groups_test = groups[test_idx]

# 5. Hitung ulang jumlah baris per grup kueri untuk parameter input LightGBM Ranker
train_group_counts = pd.Series(groups_train).value_counts().sort_index().values
test_group_counts = pd.Series(groups_test).value_counts().sort_index().values

logger.info(f"Train size: {len(X_train)} rows, Test size: {len(X_test)} rows")
logger.info(f"Train groups: {len(train_group_counts)} projects, Test groups: {len(test_group_counts)} projects")

# 6. Validasi ketat: Pastikan tidak ada proyek yang tumpang tindih
train_projects = set(np.unique(groups_train))
test_projects = set(np.unique(groups_test))
overlap = train_projects & test_projects
if overlap:
    raise ValueError(f"Project overlap between train and test: {overlap}")
logger.info("No project overlap between train and test sets. Split is clean.")


# 1. Hitung jumlah maksimum num_leaves secara dinamis berdasarkan ukuran grup terbesar
max_group_size = df.groupby("project_id").size().max()
num_leaves = min(max_group_size, 255)

# 2. Inisialisasi LGBMRanker
ranker = lgb.LGBMRanker(
    objective="lambdarank",
    boosting_type="gbdt",
    n_estimators=100,
    num_leaves=num_leaves,
    learning_rate=0.1,
    min_child_samples=10,
    random_state=RANDOM_STATE,
    verbose=-1
)

logger.info("Training LightGBM Ranker for Word2Vec...")
# 3. Fit model menggunakan array group counts yang sudah kita split di sel sebelumnya
ranker.fit(
    X_train, y_train,
    group=train_group_counts,
    eval_set=[(X_test, y_test)],
    eval_group=[test_group_counts],
    eval_metric=["ndcg"],
    callbacks=[lgb.log_evaluation(0)]
)
logger.info("Training complete.")


# ==========================================
# REKONSTRUKSI NAMA FITUR UNTUK IMPORTANCE (WORD2VEC FIXED)
# ==========================================
# Ambil semua nama kolom Word2Vec langsung dari DataFrame karena tidak ada seleksi fitur
w2v_cols = [c for c in df.columns if c.startswith("w2v_")]

# Satukan semua nama fitur sesuai urutan horizontal stack (hstack) di STEP 4 kemarin:
# Fitur Bisnis + Fitur Teks Word2Vec + Fitur Kategori (One-Hot)
all_feature_names = num_features + w2v_cols + list(encoded_type_df.columns)

# Tampilkan 10 fitur paling berpengaruh tanpa error
feature_imp_series = pd.Series(ranker.feature_importances_, index=all_feature_names)
logger.info(f"Feature importances (top 10):\n{feature_imp_series.sort_values(ascending=False).head(10)}")
from sklearn.metrics import average_precision_score, ndcg_score

logger.info("Generating predictions on test set...")
# 1. Prediksi skor relevansi menggunakan model ranker
y_pred = ranker.predict(X_test)

test_project_ids = groups_test
ndcg5_scores = []
ndcg10_scores = []
map_per_group = []

logger.info("Computing ranking metrics per project group...")
# 2. Iterasi per kelompok proyek untuk menghitung NDCG dan MAP
for pid in np.unique(test_project_ids):
    mask = test_project_ids == pid
    y_true_group = y_test[mask]
    y_pred_group = y_pred[mask]
    n = len(y_true_group)

    # Lewati proyek yang isinya cuma 1 kebutuhan karena tidak bisa diranking
    if n <= 1:
        continue

    # NDCG@5 (Diperbaiki: Proyek kecil tetap dihitung dengan k dinamis)
    k5 = min(5, n)
    ndcg5_scores.append(ndcg_score(y_true_group.reshape(1, -1), y_pred_group.reshape(1, -1), k=k5))
    
    # NDCG@10 (Diperbaiki: Proyek kecil tetap dihitung dengan k dinamis)
    k10 = min(10, n)
    ndcg10_scores.append(ndcg_score(y_true_group.reshape(1, -1), y_pred_group.reshape(1, -1), k=k10))

    # MAP (Binarisasi: top half labels dianggap relevan)
    if len(np.unique(y_true_group)) > 1:
        threshold = y_true_group.max() * 0.5
        y_bin = (y_true_group >= threshold).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            map_per_group.append(average_precision_score(y_bin, y_pred_group))

# Hitung rata-rata metrik
ndcg5 = float(np.mean(ndcg5_scores)) if ndcg5_scores else 0.0
ndcg10 = float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0
map_score = float(np.mean(map_per_group)) if map_per_group else 0.0

# 3. Hitung korelasi ranking global (Spearman & Kendall Tau)
spearman_corr, spearman_p = spearmanr(y_test, y_pred)
kendall_corr, kendall_p = kendalltau(y_test, y_pred)

# Group semua metrik ke dalam dictionary
metrics = {
    "NDCG_at_5": round(float(ndcg5), 6),
    "NDCG_at_10": round(float(ndcg10), 6),
    "MAP": round(float(map_score), 6),
    "Spearman": round(float(spearman_corr), 6),
    "Spearman_pvalue": float(spearman_p),
    "KendallTau": round(float(kendall_corr), 6),
    "KendallTau_pvalue": float(kendall_p)
}

logger.info("=== Evaluation Metrics ===")
for k, v in metrics.items():
    logger.info(f"  {k}: {v}")



# Tampilkan tabel metrik di Jupyter Notebook
display(pd.DataFrame([metrics]))
logger.info("Saving training artifacts and generated dataset locally...")

# 1. Save trained model
model_path = os.path.join(OUTPUT_DIR, "lgbm_ranker.pkl")
joblib.dump(ranker, model_path)
logger.info(f"Model saved locally: {model_path}")

# 2. Save generated dataset (Phase 2 input)
dataset_out = pd.DataFrame(X_fused, columns=all_feature_names)
dataset_out["label"] = y
dataset_out["rank"] = df["rank"].values
dataset_out["project_id"] = groups

# UBAH: Ganti nama file dari tfidf_dataset.csv menjadi w2v_dataset.csv
dataset_path = os.path.join(OUTPUT_DIR, "w2v_dataset.csv") 
dataset_out.to_csv(dataset_path, index=False)
logger.info(f"Generated dataset saved locally: {dataset_path} (shape: {dataset_out.shape})")

# 3. Save evaluation metrics
metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
logger.info(f"Metrics saved locally: {metrics_path}")

# 4. Save predictions
predictions_df = pd.DataFrame({
    "project_id": groups_test,
    "true_label": y_test,
    "predicted_score": y_pred
})
pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
predictions_df.to_csv(pred_path, index=False)
logger.info(f"Predictions saved locally: {pred_path}")

# 5. Save feature list
features_path = os.path.join(OUTPUT_DIR, "feature_list.txt")
with open(features_path, "w") as f:
    f.write("\n".join(all_feature_names))
logger.info(f"Feature list saved locally: {features_path}")
if mlflow.active_run():
    mlflow.end_run()

try:
    with mlflow.start_run(run_name="Word2Vec_LightGBM_Ranker_Experiment") as run:
        # Log parameters
        mlflow.log_param("text_representation", "Word2Vec")
        mlflow.log_param("feature_selection_method", globals().get("FEATURE_SELECTION_METHOD", "None"))
        mlflow.log_param("w2v_features_count", X_w2v.shape[1])
        mlflow.log_param("total_fused_features", X_fused.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("num_train_groups", len(train_group_counts))
        mlflow.log_param("num_test_groups", len(test_group_counts))
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size_ratio", globals().get("TEST_SIZE", 0.2))
        mlflow.log_param("ranker_objective", "lambdarank")
        mlflow.log_param("ranker_n_estimators", 100)
        mlflow.log_param("ranker_num_leaves", num_leaves)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log artifacts
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
        mlflow.log_artifact(features_path, artifact_path="features")
        mlflow.log_artifact(dataset_path, artifact_path="dataset")
        mlflow.log_artifact(pred_path, artifact_path="predictions")

        logger.info(f"MLflow run ID: {run.info.run_id}")
        logger.info("MLflow experiment logged successfully.")
except Exception as e:
    logger.warning(f"MLflow logging error: {e}. Experiment execution & local artifacts completed successfully.")


# 1. Siapkan DataFrame rangkuman metrik agar rapi saat di-display
summary_data = {
    "Metric": ["NDCG@5", "NDCG@10", "MAP", "Spearman rho", "Spearman p-value", "Kendall tau", "Kendall tau p-value"],
    "Value": [
        metrics["NDCG_at_5"],
        metrics["NDCG_at_10"],
        metrics["MAP"],
        metrics["Spearman"],
        metrics["Spearman_pvalue"],
        metrics["KendallTau"],
        metrics["KendallTau_pvalue"]
    ]
}
summary_df = pd.DataFrame(summary_data)

# 2. Cetak log parameter eksperimen Word2Vec ke konsol/terminal
logger.info("=== Experiment Summary ===")
logger.info("Text Representation: Word2Vec")  # <-- UBAH: Set ke Word2Vec
logger.info("Feature Selection Method: None (Skipped)")
logger.info(f"Word2Vec Text Dimension: {X_w2v.shape[1]}")  # <-- UBAH: Panggil X_w2v hasil STEP 3
logger.info(f"Train size: {len(X_train)} rows, Test size: {len(X_test)} rows")
logger.info(f"Train groups: {len(train_group_counts)} projects, Test groups: {len(test_group_counts)} projects")
logger.info(f"Total Final Features (Fused): {X_fused.shape[1]}")
logger.info(f"\n{summary_df.to_string(index=False)}")

# 3. Tampilkan tabel metrik di Jupyter notebook
display(summary_df)