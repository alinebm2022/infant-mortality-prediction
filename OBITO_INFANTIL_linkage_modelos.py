"""Projeto de portfolio: predição de óbito infantil usando linkage determinístico/probabilístico.

Modelos: Regressão Logística (RLM), MLP e XGBoost.
CV: Stratified 5-fold no conjunto de desenvolvimento; teste holdout separado.
Balanceamento: nenhum nesta versão principal (comparação de 6 configurações).

ATENÇÃO: linkage probabilístico usa EM simplificado para estimar parâmetros m/u
(Fellegi-Sunter) em pares candidatos bloqueados por data de nascimento. Avalie
manualmente a qualidade do linkage antes de interpretar resultados epidemiológicos.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, average_precision_score,
                            balanced_accuracy_score, precision_recall_curve)
from xgboost import XGBClassifier
import statsmodels.api as sm



# ============================ CONFIG ============================
INPUT_DIR = Path(r"C:\Users\SEU_USUARIO\Documents\dados")  # altere
SINASC_FILE = "DNRJ2023.csv"
SIM_2023_FILE = "DOINF23.csv"
SIM_2024_FILE = "DOINF24.csv"
OUTPUT_FILE = "OBITO_INFANTIL_linkage_modelos.xlsx"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MIN_RECALL = 0.65
PROB_LINK_THRESHOLD = 0.50  # posterior probability for FS linkage


# ============================ DATA ============================
def date_from_numeric(series):
    return pd.to_datetime(series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8),
                          format="%d%m%Y", errors="coerce")



def load_data():
    vivos = pd.read_csv(INPUT_DIR / SINASC_FILE, low_memory=False)
    mortes = pd.concat([pd.read_csv(INPUT_DIR / SIM_2023_FILE, low_memory=False),
                       pd.read_csv(INPUT_DIR / SIM_2024_FILE, low_memory=False)], ignore_index=True)
    birth_cols = ["CODMUNNASC", "APGAR5", "LOCNASC", "PESO", "SEXO", "PARTO", "DTNASC",
                 "IDADEMAE", "ESTCIVMAE", "CODMUNRES", "GRAVIDEZ", "IDANOMAL", "ESCMAE2010",
                 "QTDGESTANT", "KOTELCHUCK", "TPAPRESENT", "QTDPARTNOR", "QTDPARTCES",
                 "SEMAGESTAC", "STTRABPART", "RACACOR"]
    death_cols = ["PESO", "SEXO", "RACACOR", "ESCMAE2010", "PARTO", "DTNASC", "DTOBITO",
                 "CODMUNRES", "CODMUNNATU", "GRAVIDEZ", "LOCOCOR"]
    vivos = vivos[birth_cols].copy()
    mortes = mortes[death_cols].copy().rename(columns={"CODMUNNATU": "CODMUNNASC"})
    vivos["DTNASC"] = date_from_numeric(vivos["DTNASC"])
    mortes["DTNASC"] = date_from_numeric(mortes["DTNASC"])
    mortes["DTOBITO"] = date_from_numeric(mortes["DTOBITO"])
    deaths_link = mortes[["DTNASC", "SEXO", "PESO", "PARTO", "CODMUNNASC", "DTOBITO"]].copy()
    return vivos, deaths_link


# ============================ LINKAGE ============================
def deterministic_linkage(births, deaths):
    # preserve duplicate handling transparently; validate is deliberately not restrictive
    out = births.merge(deaths, on=["DTNASC", "SEXO", "PESO", "PARTO", "CODMUNNASC"],
                      how="left", indicator=True)
    return out



def _em_fellegi_sunter(comp, max_iter=100, tol=1e-6):
    """Estimate m/u and match prior with a binary comparison matrix using EM.
    This is an unsupervised approximation; EM may converge to local optima.
    """
    comp = np.asarray(comp, dtype=float)
    n, p = comp.shape
    if n < 10:
        # fallback to smoothed agreement-based weights if too few candidate pairs
        m = np.clip(comp.mean(axis=0), 0.70, 0.99)
        u = np.clip(comp.mean(axis=0) * 0.20, 0.01, 0.50)
        prior = 0.01
        return m, u, prior
    prior = 0.02
    m = np.clip(comp.mean(axis=0) + 0.10, 0.50, 0.99)
    u = np.clip(comp.mean(axis=0) - 0.20, 0.01, 0.80)
    for _ in range(max_iter):
        # log likelihoods for each pair under match and non-match classes
        log_m = np.log(np.clip(m, 1e-6, 1-1e-6))
        log_1m = np.log(np.clip(1-m, 1e-6, 1-1e-6))
        log_u = np.log(np.clip(u, 1e-6, 1-1e-6))
        log_1u = np.log(np.clip(1-u, 1e-6, 1-1e-6))
        lm = comp @ log_m + (1-comp) @ log_1m + np.log(prior)
        lu = comp @ log_u + (1-comp) @ log_1u + np.log(1-prior)
        mx = np.maximum(lm, lu)
        post = np.exp(lm-mx) / (np.exp(lm-mx) + np.exp(lu-mx))
        new_prior = np.clip(post.mean(), 1e-5, 0.20)
        new_m = (post[:, None] * comp).sum(axis=0) / max(post.sum(), 1e-8)
        new_u = ((1-post)[:, None] * comp).sum(axis=0) / max((1-post).sum(), 1e-8)
        new_m = np.clip(new_m, 1e-4, 1-1e-4)
        new_u = np.clip(new_u, 1e-4, 1-1e-4)
        delta = max(np.max(np.abs(new_m-m)), np.max(np.abs(new_u-u)), abs(new_prior-prior))
        m, u, prior = new_m, new_u, new_prior
        if delta < tol:
            break
    return m, u, prior



def probabilistic_linkage(births, deaths, threshold=PROB_LINK_THRESHOLD):
    """Date-of-birth blocking + EM-estimated Fellegi-Sunter posterior.
    Comparisons: municipality, sex, birth weight, delivery type. One-to-one greedy assignment.
    """
    b = births.reset_index(drop=False).rename(columns={"index": "birth_id"}).copy()
    d = deaths.reset_index(drop=False).rename(columns={"index": "death_id"}).copy()
    # Explicit IDs in case original indices are not named index.
    b["birth_id"] = np.arange(len(b)); d["death_id"] = np.arange(len(d))
    fields = ["CODMUNNASC", "SEXO", "PESO", "PARTO"]
    pair_chunks = []
    for date, bg in b.groupby("DTNASC", dropna=True):
        dg = d[d["DTNASC"] == date]
        if bg.empty or dg.empty:
            continue
        # Candidate pair generation (blocking) by exact birth date.
        left = bg[["birth_id"] + fields].copy()
        right = dg[["death_id"] + fields].copy()
        left["_key"] = 1; right["_key"] = 1
        pairs = left.merge(right, on="_key", suffixes=("_b", "_d")).drop(columns="_key")
        comp = np.column_stack([pairs[f + "_b"].eq(pairs[f + "_d"]).fillna(False).astype(int) for f in fields])
        comp = np.asarray(comp, dtype=float)
        m, u, prior = _em_fellegi_sunter(comp)
        log_odds = np.log(prior/(1-prior))
        for j in range(len(fields)):
            log_odds += np.where(comp[:, j] == 1, np.log(m[j]/u[j]), np.log((1-m[j])/(1-u[j])))
        post = 1/(1+np.exp(-np.clip(log_odds, -50, 50)))
        pairs["posterior_match"] = post
        pair_chunks.append(pairs[["birth_id", "death_id", "posterior_match"]])
    if not pair_chunks:
        raise ValueError("Probabilistic linkage: nenhum par candidato. Verifique DTNASC.")
    all_pairs = pd.concat(pair_chunks, ignore_index=True)
    accepted = all_pairs[all_pairs.posterior_match >= threshold].sort_values("posterior_match", ascending=False)
    # Greedy one-to-one assignment; highest posterior pairs first.
    accepted = accepted.drop_duplicates("death_id").drop_duplicates("birth_id")
    out = b.copy()
    out["DTOBITO"] = pd.NaT
    out["LINK_SCORE"] = np.nan
    matched = accepted.merge(d[["death_id", "DTOBITO"]], on="death_id", how="left")
    out.loc[matched.birth_id, "DTOBITO"] = matched.set_index("birth_id")["DTOBITO"]
    out.loc[matched.birth_id, "LINK_SCORE"] = matched.set_index("birth_id")["posterior_match"]
    out["_merge"] = np.where(out["DTOBITO"].notna(), "both", "left_only")
    return out


# ============================ FEATURES / TARGET ============================
def prepare_model_data(linked):
    df = linked.copy()
    # Target: death before age one. Linkage is restricted to deaths from the supplied SIM files.
    age_days = (df["DTOBITO"] - df["DTNASC"]).dt.days
    df["OBITO_INFANTIL"] = ((df["_merge"] == "both") & age_days.between(0, 364)).astype(int)
    # Retain predictors available at birth. Linkage-only fields remain excluded from predictors.
    df["PARIDADE_NUM"] = df["QTDPARTNOR"] + df["QTDPARTCES"]
    invalid_parity = [100,101,102,103,106,112,198,30,31,26,24,21,38,25,23,20,44,39,52]
    df = df[~df["PARIDADE_NUM"].isin(invalid_parity)]
    df = df[(df["GRAVIDEZ"] == 1) & (df["LOCNASC"] == 1) & (df["IDANOMAL"] == 2)]
    df.drop(columns=["QTDPARTNOR", "QTDPARTCES", "QTDGESTANT", "IDANOMAL", "LOCNASC", "GRAVIDEZ"], inplace=True, errors="ignore")
    # Remove invalid codes and missing values from fields used in the study.
    df = df.dropna().copy()
    integer_cols = ["APGAR5", "PESO", "PARTO", "ESTCIVMAE", "ESCMAE2010", "TPAPRESENT",
                    "SEMAGESTAC", "STTRABPART", "RACACOR", "PARIDADE_NUM"]
    df[integer_cols] = df[integer_cols].astype(int)
    df["APGAR5"] = np.where(df["APGAR5"] < 7, "Ruim", "Bom")
    df["Idade_materna"] = np.select([df.IDADEMAE < 20, df.IDADEMAE.between(20,34), df.IDADEMAE >= 35],
                                    ["Adolescente", "Adulta Jovem", "Adulta Madura"], default="Desconhecida")
    df["PESO"] = np.select([df.PESO <= 1500, df.PESO.between(1501,2499), df.PESO.between(2500,3999), df.PESO >= 4000],
                            ["Baixo peso extremo", "Baixo Peso", "Normal", "Macrossomia"], default="Desconhecido")
    df["SEXO"] = df["SEXO"].map({1:"Masculino", 2:"Feminino"}).fillna("Desconhecido")
    df["PARTO"] = df["PARTO"].map({1:"Vaginal", 2:"Cesárea"}).fillna("Desconhecido")
    df["KOTELCHUCK"] = df["KOTELCHUCK"].map({1:"Nao fez",2:"Inadequado",3:"Intermediario",4:"Adequado",5:"Mais que adequado"}).fillna("Desconhecido")
    df["Estado_civil"] = np.where(df["ESTCIVMAE"].isin([1,3,4]), "Sem companheiro", "Com companheiro")
    df["Escolaridade_materna"] = np.select([df.ESCMAE2010==0, df.ESCMAE2010.isin([1,2]), df.ESCMAE2010==3, df.ESCMAE2010.isin([4,5])],
                                          ["Sem escolaridade","Fundamental","Medio","Superior"], default="Desconhecido")
    df["Paridade"] = np.select([df.PARIDADE_NUM==0, df.PARIDADE_NUM.between(1,3), df.PARIDADE_NUM>=4],
                               ["Nulípara","Baixa multiparidade","Alta multiparidade"], default="Desconhecido")
    df["Idade_gestacional"] = np.select([df.SEMAGESTAC<32, df.SEMAGESTAC.between(32,36), df.SEMAGESTAC.between(37,41), df.SEMAGESTAC>=42],
                                        ["Muito Prematuro","Prematuro","A Termo","Pos-termo"], default="Desconhecido")
    df["Apresentacao"] = np.where(df.TPAPRESENT==1, "Cefalico", np.where(df.TPAPRESENT.isin([2,3]), "Nao-cefalico", "Desconhecido"))
    df["Inducao"] = df["STTRABPART"].map({1:"Induzido",2:"Nao-induzido"}).fillna("Desconhecido")
    df["RACACOR"] = np.where(df["RACACOR"]==1, "Branca", "Nao-branca")
    df["Peregrinacao"] = np.where(df["CODMUNNASC"]==df["CODMUNRES"], "Não", "Sim")
    # Remove raw columns, death date, linkage indicators, and identifiers.
    drop_cols = ["DTNASC", "DTOBITO", "_merge", "LINK_SCORE", "IDADEMAE", "ESTCIVMAE", "ESCMAE2010",
                 "PARIDADE_NUM", "SEMAGESTAC", "TPAPRESENT", "STTRABPART", "CODMUNNASC", "CODMUNRES"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    # Outcome is numeric 0/1; avoid predictors derived from outcome.
    X = df.drop(columns="OBITO_INFANTIL")
    y = df["OBITO_INFANTIL"].astype(int)
    return X, y


# ============================ PREPROCESSORS / MODELS ============================
def build_preprocessor(X):
    categorical = list(X.columns)
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True))])
    return ColumnTransformer([("categorical", cat_pipe, categorical)], remainder="drop")


def build_model(model_name, preprocessor):
    if model_name == "RLM":
        classifier = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=None, random_state=RANDOM_STATE)
    elif model_name == "MLP":
        classifier = MLPClassifier(hidden_layer_sizes=(22, 11), activation="relu", solver="adam",
                                 max_iter=500, early_stopping=True, validation_fraction=0.15,
                                 random_state=RANDOM_STATE)
    elif model_name == "XGBoost":
        classifier = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE,
                                 n_jobs=-1, tree_method="hist")
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])



def choose_threshold(y_true, probs, min_recall=MIN_RECALL):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    precision, recall = precision[:-1], recall[:-1]
    valid = recall >= min_recall
    if valid.any():
        ix = np.argmax(precision[valid])
        return float(thresholds[valid][ix]), "recall_minimo_atendido"
    if len(thresholds):
        return float(thresholds[np.argmax(recall)]), "fallback_max_recall"
    return 0.5, "threshold_padrao"



def metrics_row(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    specificity = tn/(tn+fp) if tn+fp else np.nan
    return {"ROC_AUC": roc_auc_score(y, prob) if len(np.unique(y))>1 else np.nan,
            "PR_AUC": average_precision_score(y, prob), "Acuracia": accuracy_score(y,pred),
            "Balanced_Accuracy": balanced_accuracy_score(y,pred), "Precisao": precision_score(y,pred,zero_division=0),
            "Recall_Sensibilidade": recall_score(y,pred,zero_division=0), "F1": f1_score(y,pred,zero_division=0),
            "Especificidade": specificity, "TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp),
            "Threshold":float(threshold)}


# ============================ CV AND HOLDOUT ============================
def run_cv(X_dev, y_dev, model_name, n_splits=CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows=[]
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
        X_tr, X_va = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_va = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        pipe = build_model(model_name, build_preprocessor(X_tr))
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:,1]
        # CV metrics use the fixed 0.50 threshold to avoid optimizing and evaluating
        # the threshold on the same fold. Threshold tuning is reserved for validation.
        thr = 0.50
        row = metrics_row(y_va, p, thr)
        row.update({"Modelo":model_name,"Fold":fold,"Threshold_status":"threshold_fixo_0.50_CV"})
        fold_rows.append(row)
    cv_df=pd.DataFrame(fold_rows)
    numeric_metrics=[c for c in cv_df.columns if c in ["ROC_AUC","PR_AUC","Acuracia","Balanced_Accuracy","Precisao","Recall_Sensibilidade","F1","Especificidade"]]
    summary=[]
    for metric in numeric_metrics:
        summary.append({"Modelo":model_name,"Metrica":metric,"CV_Media":cv_df[metric].mean(),"CV_DP":cv_df[metric].std(ddof=1)})
    return cv_df, pd.DataFrame(summary)


def fit_holdout_and_explain(X_train, X_val, X_test, y_train, y_val, y_test, model_name, linkage_label):
    # fit final model using train split, choose threshold on validation, evaluate once on test.
    pipe=build_model(model_name, build_preprocessor(X_train))
    pipe.fit(X_train, y_train)
    val_prob=pipe.predict_proba(X_val)[:,1]
    threshold, threshold_status=choose_threshold(y_val, val_prob)
    test_prob=pipe.predict_proba(X_test)[:,1]
    row=metrics_row(y_test, test_prob, threshold)
    row.update({"Linkage":linkage_label,"Modelo":model_name,"Threshold_status":threshold_status,
               "N_train":len(y_train),"N_val":len(y_val),"N_test":len(y_test),
               "Prevalencia_train":float(y_train.mean()),"Prevalencia_test":float(y_test.mean())})
    odds=[]; pfi=[]
    if model_name=="RLM":
        # Statsmodels logistic for coefficient table and adjusted ORs on encoded features.
        prep=pipe.named_steps["preprocessor"]
        Xtr_enc=prep.transform(X_train)
        Xte_enc=prep.transform(X_test)
        names=prep.get_feature_names_out()
        # statsmodels GLM binomial provides estimates, standard errors and Wald CIs.
        try:
            X_dense = Xtr_enc.toarray() if hasattr(Xtr_enc, "toarray") else np.asarray(Xtr_enc)
            sm_model=sm.GLM(y_train, sm.add_constant(X_dense, has_constant="add"), family=sm.families.Binomial()).fit()
            params=np.asarray(sm_model.params); conf=np.asarray(sm_model.conf_int()); pvals=np.asarray(sm_model.pvalues)
            for i, name in enumerate(["Intercept"]+list(names)):
                beta=params[i]; lo,hi=conf[i]
                odds.append({"Linkage":linkage_label,"Modelo":model_name,"Term":name,"Coeficiente":beta,
                            "OR":np.exp(np.clip(beta,-30,30)),"OR_CI95_lower":np.exp(np.clip(lo,-30,30)),
                            "OR_CI95_upper":np.exp(np.clip(hi,-30,30)),"P_value_Wald":pvals[i]})
        except Exception as e:
            warnings.warn(f"OR statsmodels falhou para {linkage_label}: {e}")
    elif model_name=="MLP":
        try:
            # Feature permutation on held-out test, at ROC-AUC metric (threshold-independent).
            pi=permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=5,
                                    random_state=RANDOM_STATE, n_jobs=-1)
            pfi=pd.DataFrame({"Linkage":linkage_label,"Modelo":model_name,"Variavel":X_test.columns,
                            "Importance_mean":pi.importances_mean,"Importance_std":pi.importances_std})
            pfi=pfi.sort_values("Importance_mean",ascending=False)
        except Exception as e:
            warnings.warn(f"Permutation Importance falhou: {e}")
    return row, pd.DataFrame(odds), pfi, pipe


# ============================ RUN PER LINKAGE ============================
def run_linkage_experiment(linked, linkage_label):
    X,y=prepare_model_data(linked)
    # Stratified holdout, then train/validation split from development data.
    X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=0.20,stratify=y_dev,random_state=RANDOM_STATE)
    holdout_rows=[]; cv_fold_frames=[]; cv_summary_frames=[]; odds_frames=[]; pfi_frames=[]
    for model_name in ["RLM","MLP","XGBoost"]:
        print(f"\n[{linkage_label}] {model_name}: Cross-validation {CV_FOLDS}-fold")
        cv_folds, cv_summary=run_cv(X_train,y_train,model_name)
        cv_folds.insert(0,"Linkage",linkage_label)
        cv_fold_frames.append(cv_folds); cv_summary.insert(0,"Linkage",linkage_label); cv_summary_frames.append(cv_summary)
        print(f"[{linkage_label}] {model_name}: avaliação holdout")
        row, odds, pfi, model=fit_holdout_and_explain(X_train,X_val,X_test,y_train,y_val,y_test,model_name,linkage_label)
        holdout_rows.append(row)
        if not odds.empty: odds_frames.append(odds)
        if not pfi.empty: pfi_frames.append(pfi)
    return (pd.DataFrame(holdout_rows), pd.concat(cv_fold_frames,ignore_index=True),
            pd.concat(cv_summary_frames,ignore_index=True),
            pd.concat(odds_frames,ignore_index=True) if odds_frames else pd.DataFrame(),
            pd.concat(pfi_frames,ignore_index=True) if pfi_frames else pd.DataFrame())


# ============================ MAIN ============================
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    births,deaths=load_data()
    print(f"SINASC: {births.shape} | SIM: {deaths.shape}")
    det=deterministic_linkage(births,deaths)
    prob=probabilistic_linkage(births,deaths)
    outputs={}
    for label,linked in [("Deterministico",det),("Probabilistico",prob)]:
        outputs[label]=run_linkage_experiment(linked,label)
    metrics=pd.concat([outputs[k][0] for k in outputs],ignore_index=True)
    cv_folds=pd.concat([outputs[k][1] for k in outputs],ignore_index=True)
    cv_summary=pd.concat([outputs[k][2] for k in outputs],ignore_index=True)
    odds=pd.concat([outputs[k][3] for k in outputs if not outputs[k][3].empty],ignore_index=True) if any(not outputs[k][3].empty for k in outputs) else pd.DataFrame()
    pfi=pd.concat([outputs[k][4] for k in outputs if not outputs[k][4].empty],ignore_index=True) if any(not outputs[k][4].empty for k in outputs) else pd.DataFrame()
    linkage_summary=pd.DataFrame({"Linkage":["Deterministico","Probabilistico"],
                                "N_registros_linked":[int((det["_merge"]=="both").sum()),int((prob["_merge"]=="both").sum())],
                                "N_birth_records":[len(det),len(prob)]})
    with pd.ExcelWriter(OUTPUT_FILE,engine="openpyxl") as writer:
        metrics.to_excel(writer,sheet_name="Holdout_metrics",index=False)
        cv_folds.to_excel(writer,sheet_name="CV_5fold_por_fold",index=False)
        cv_summary.to_excel(writer,sheet_name="CV_media_DP",index=False)
        if not odds.empty: odds.to_excel(writer,sheet_name="RLM_OR_IC95",index=False)
        if not pfi.empty: pfi.to_excel(writer,sheet_name="MLP_Permutation_Importance",index=False)
        linkage_summary.to_excel(writer,sheet_name="Resumo_linkage",index=False)
    print("\nExcel criado:", OUTPUT_FILE)
    print(metrics.to_string(index=False))

if __name__=="__main__":
    main()
