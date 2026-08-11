import os


PKG_DIR = os.path.dirname(os.path.abspath(__file__))


SEED = 42


SAMPLE = None


DATASETS = ["cnn_dailymail", "xsum"]


DATASET_CONFIGS = {

    "cnn_dailymail": {"path": "abisee/cnn_dailymail", "name": "3.0.0"},

    "xsum":          {"path": "EdinburghNLP/xsum"},

}

FIELD_MAP = {

    "cnn_dailymail": ("article",  "highlights"),

    "xsum":          ("document", "summary"),

}

DATASETS_WITH_DATELINES = {"cnn_dailymail"}


FULL_TEST_SIZES = {"cnn_dailymail": 11490, "xsum": 11334}

FULL_TRAIN_SIZES = {"cnn_dailymail": 287113, "xsum": 204045}


SUMMARIES_DIR = os.path.join(PKG_DIR, "summaries")

RESULTS_DIR = os.path.join(PKG_DIR, "results")

METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")

MODELS_DIR = os.path.join(PKG_DIR, "models")

FEATURES_DIR = os.path.join(PKG_DIR, "data", "features")


def output_filename(label: str, dataset: str) -> str:

    return f"{label}_{dataset}_summaries.jsonl"


def model_dir(models_root: str, dataset: str, name: str) -> str:

    return os.path.join(models_root, dataset, name)


MODELS = [

    {"name": "logreg", "label": "ML-LogReg"},

    {"name": "nb",     "label": "ML-NB"},

    {"name": "xgb",    "label": "ML-XGB"},

]

MODEL_NAMES = [m["name"] for m in MODELS]

LABELS = {m["name"]: m["label"] for m in MODELS}


USES_SPARSE = {"logreg": True, "nb": True, "xgb": False}


HASH_DIM = 2 ** 18

CHUNK_DOCS = 5000


ORACLE_MAX_K = {"cnn_dailymail": 5, "xsum": 2}


K_GRID = [1, 2, 3, 4, 5]

K_CAP = {"cnn_dailymail": 5, "xsum": 3}

FALLBACK = {

    "cnn_dailymail": {"k": 3, "blocking": True},

    "xsum": {"k": 1, "blocking": False},

}


COMPOSITE_METRICS = ["rougeL", "meteor", "bleu"]


VAL_DOCS = None


WORD_BUDGETS = {"cnn_dailymail": [45, 55, 70], "xsum": [20, 30]}

MMR_LAMBDAS = [0.3, 0.5, 0.7]

FALLBACK_SELECTION = {

    "cnn_dailymail": {"mode": "topk", "k": 3, "redundancy": "block"},

    "xsum": {"mode": "topk", "k": 1, "redundancy": "none"},

}


def selection_policy_grid(dataset: str) -> list[dict]:


    lengths = [{"mode": "topk", "k": k} for k in K_GRID if k <= K_CAP[dataset]]

    lengths += [{"mode": "budget", "budget": b} for b in WORD_BUDGETS[dataset]]

    redundancies = [{"redundancy": "none"}, {"redundancy": "block"}]

    redundancies += [{"redundancy": "mmr", "lambda": lam} for lam in MMR_LAMBDAS]

    return [{**length, **red} for length in lengths for red in redundancies]


HPARAM_SUBSAMPLE = None

HPARAM_CV_FOLDS = 3

HPARAM_SCORING = "average_precision"

HPARAM_N_ITER = 20


HPARAM_N_JOBS = 4


HPARAM_GRIDS = {

    "logreg": {

        "alpha": [1e-5, 1e-4, 1e-3],

        "penalty": ["l2", "elasticnet"],

        "l1_ratio": [0.0, 0.15],

    },

    "nb": {

        "alpha": [0.1, 0.5, 1.0, 2.0],

    },

    "xgb": {

        "n_estimators": [200, 400],

        "max_depth": [4, 6, 8],

        "learning_rate": [0.05, 0.1, 0.2],

        "subsample": [0.8, 1.0],

        "colsample_bytree": [0.8, 1.0],

        "reg_lambda": [0.0, 1.0],

    },

}
