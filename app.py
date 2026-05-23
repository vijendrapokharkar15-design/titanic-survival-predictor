import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# load model
with open("model/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/features.pkl", "rb") as f:
    features = pickle.load(f)

# page config
st.set_page_config(
    page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide"
)

st.title("🚢 Titanic Survival Predictor")
st.write("Built with Phase 2 techniques — Random Forest + SHAP + Feature Engineering")
st.divider()

# tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict", "📊 Data Insights", "🤖 Model Comparison", "📁 Batch Predict"]
)


# helper — build features from inputs
def build_features(pclass, sex, age, fare, sibsp, parch, has_cabin, title):
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_log = np.log1p(fare)
    sex_encoded = 1 if sex == "Female" else 0
    return {
        "Pclass": pclass,
        "Age": age,
        "FareLog": fare_log,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "HasCabin": int(has_cabin),
        "Sex_encoded": sex_encoded,
        "Title_Mr": 1 if title == "Mr" else 0,
        "Title_Mrs": 1 if title == "Mrs" else 0,
        "Title_Miss": 1 if title == "Miss" else 0,
        "Title_Master": 1 if title == "Master" else 0,
    }


# ── TAB 1: PREDICT ──────────────────────────────────────
with tab1:
    col_input, col_output = st.columns([1, 2])

    with col_input:
        st.subheader("Passenger Details")
        pclass = st.selectbox(
            "Class",
            [1, 2, 3],
            format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class",
        )
        sex = st.radio("Sex", ["Male", "Female"])
        age = st.slider("Age", 1, 80, 30)
        fare = st.slider("Fare (£)", 0, 520, 32)
        sibsp = st.number_input("Siblings/Spouses", 0, 8, 0)
        parch = st.number_input("Parents/Children", 0, 6, 0)
        has_cabin = st.checkbox("Has cabin number?")
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])

    with col_output:
        st.subheader("Prediction")
        input_data = build_features(
            pclass, sex, age, fare, sibsp, parch, has_cabin, title
        )
        input_df = pd.DataFrame([input_data])
        prob = model.predict_proba(input_df)[0][1]

        # big metric
        if prob >= 0.5:
            st.success(f"## Survived ✅  —  {prob*100:.1f}% probability")
        else:
            st.error(f"## Died ❌  —  {prob*100:.1f}% probability")

        st.progress(float(prob))
        st.divider()

        # metrics row
        family_size = sibsp + parch + 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Class", f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}"
        )
        c2.metric("Age", age)
        c3.metric("Fare", f"£{fare}")
        c4.metric("Family", family_size)
        st.divider()

        # SHAP
        st.subheader("Why this prediction?")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_df)
            shap_vals = shap_values[:, :, 1].values[0]
            shap_df = pd.DataFrame(
                {"Feature": features, "SHAP": shap_vals, "Value": input_df.values[0]}
            ).sort_values("SHAP", key=abs, ascending=False)

            for _, row in shap_df.iterrows():
                icon = "🟢" if row["SHAP"] > 0 else "🔴"
                direction = "helps" if row["SHAP"] > 0 else "hurts"
                st.write(
                    f"{icon} **{row['Feature']}** = {row['Value']:.2f} "
                    f"→ {row['SHAP']:+.4f} ({direction} survival)"
                )
        except:
            st.write("SHAP unavailable")

# ── TAB 2: DATA INSIGHTS ────────────────────────────────
with tab2:
    st.subheader("Titanic Dataset Insights")

    df = pd.read_csv("titanic.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", 891)
    col2.metric("Survived", 342)
    col3.metric("Survival Rate", "38.4%")
    col4.metric("Features Used", 11)
    st.divider()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes.flat:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # survival by sex
    surv_sex = df.groupby("Sex")["Survived"].mean()
    axes[0, 0].bar(
        surv_sex.index, surv_sex.values, color=["#00b4d8", "#ff6b9d"], edgecolor="black"
    )
    axes[0, 0].set_title("Survival Rate by Sex")
    axes[0, 0].set_ylabel("Survival Rate")
    for i, v in enumerate(surv_sex.values):
        axes[0, 0].text(i, v + 0.01, f"{v:.1%}", ha="center", color="white")

    # survival by class
    surv_class = df.groupby("Pclass")["Survived"].mean()
    axes[0, 1].bar(
        ["1st", "2nd", "3rd"],
        surv_class.values,
        color=["#ffd700", "#c0c0c0", "#cd7f32"],
        edgecolor="black",
    )
    axes[0, 1].set_title("Survival Rate by Class")
    axes[0, 1].set_ylabel("Survival Rate")
    for i, v in enumerate(surv_class.values):
        axes[0, 1].text(i, v + 0.01, f"{v:.1%}", ha="center", color="white")

    # age distribution
    axes[1, 0].hist(
        df[df["Survived"] == 1]["Age"],
        bins=20,
        alpha=0.7,
        color="#00b4d8",
        label="Survived",
    )
    axes[1, 0].hist(
        df[df["Survived"] == 0]["Age"],
        bins=20,
        alpha=0.7,
        color="#ff6b9d",
        label="Died",
    )
    axes[1, 0].set_title("Age Distribution by Survival")
    axes[1, 0].set_xlabel("Age")
    axes[1, 0].legend()

    # family size
    surv_fam = df.groupby("FamilySize")["Survived"].mean()
    axes[1, 1].plot(surv_fam.index, surv_fam.values, "o-", color="#00b4d8", linewidth=2)
    axes[1, 1].set_title("Survival Rate by Family Size")
    axes[1, 1].set_xlabel("Family Size")
    axes[1, 1].set_ylabel("Survival Rate")
    axes[1, 1].axhline(0.384, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 3: MODEL COMPARISON ─────────────────────────────
with tab3:
    st.subheader("Model Comparison — RF vs LR vs GB")
    st.write("Training all three models on the same data...")

    @st.cache_data
    def train_all_models():
        df = pd.read_csv("titanic.csv")
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna("S")
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["FareLog"] = np.log1p(df["Fare"])
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Sex_encoded"] = (df["Sex"] == "female").astype(int)
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_map = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
        df["Title"] = df["Title"].map(title_map).fillna("Other")
        for t in ["Mr", "Mrs", "Miss", "Master"]:
            df[f"Title_{t}"] = (df["Title"] == t).astype(int)
        feats = [
            "Pclass",
            "Age",
            "FareLog",
            "FamilySize",
            "IsAlone",
            "HasCabin",
            "Sex_encoded",
            "Title_Mr",
            "Title_Mrs",
            "Title_Miss",
            "Title_Master",
        ]
        X = df[feats]
        y = df["Survived"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        results = {}
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, min_samples_leaf=4, random_state=42
            ),
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(random_state=42, max_iter=1000)),
                ]
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            ),
        }
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            results[name] = {
                "Accuracy": round(accuracy_score(y_te, m.predict(X_te)), 4),
                "AUC": round(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]), 4),
            }
        return results

    with st.spinner("Training models..."):
        results = train_all_models()

    res_df = pd.DataFrame(results).T
    st.dataframe(res_df, use_container_width=True)

    fig2, ax = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    colors = ["#00b4d8", "#ff6b9d", "#ffd700"]
    bars = ax.bar(
        results.keys(),
        [r["AUC"] for r in results.values()],
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )
    ax.set_title("Test AUC Comparison", color="white")
    ax.set_ylabel("AUC Score", color="white")
    ax.set_ylim(0.80, 0.90)
    for bar, (name, r) in zip(bars, results.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{r['AUC']:.4f}",
            ha="center",
            color="white",
            fontsize=10,
        )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── TAB 4: BATCH PREDICT ────────────────────────────────
with tab4:
    st.subheader("Batch Predictions — Upload a CSV")
    st.write(
        "Upload a CSV with passenger data to get predictions for multiple passengers at once."
    )

    st.markdown(
        "**Required columns:** `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch`, `Cabin`, `Name`"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} passengers")

        # engineer features
        batch_df["Age"] = batch_df["Age"].fillna(batch_df["Age"].median())
        batch_df["Fare"] = batch_df["Fare"].fillna(batch_df["Fare"].median())
        batch_df["FamilySize"] = batch_df["SibSp"] + batch_df["Parch"] + 1
        batch_df["IsAlone"] = (batch_df["FamilySize"] == 1).astype(int)
        batch_df["FareLog"] = np.log1p(batch_df["Fare"])
        batch_df["HasCabin"] = batch_df["Cabin"].notna().astype(int)
        batch_df["Sex_encoded"] = (batch_df["Sex"] == "female").astype(int)
        batch_df["Title"] = batch_df["Name"].str.extract(
            r" ([A-Za-z]+)\.", expand=False
        )
        title_map = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master"}
        batch_df["Title"] = batch_df["Title"].map(title_map).fillna("Other")
        for t in ["Mr", "Mrs", "Miss", "Master"]:
            batch_df[f"Title_{t}"] = (batch_df["Title"] == t).astype(int)

        X_batch = batch_df[features]
        probs = model.predict_proba(X_batch)[:, 1]
        preds = (probs >= 0.5).astype(int)

        batch_df["Survival_Prob"] = (probs * 100).round(1)
        batch_df["Prediction"] = ["Survived ✅" if p == 1 else "Died ❌" for p in preds]

        st.dataframe(
            batch_df[
                ["Name", "Pclass", "Sex", "Age", "Fare", "Survival_Prob", "Prediction"]
            ].head(20),
            use_container_width=True,
        )

        survived_count = preds.sum()
        st.metric("Predicted survivors", f"{survived_count}/{len(batch_df)}")
        st.download_button(
            "Download predictions CSV",
            batch_df[
                ["Name", "Pclass", "Sex", "Age", "Survival_Prob", "Prediction"]
            ].to_csv(index=False),
            "predictions.csv",
            "text/csv",
        )
    else:
        st.info(
            "Upload titanic.csv from your project folder to test batch predictions!"
        )
        st.caption("Tip: use the same titanic.csv from C:\\DS-AI-75d\\titanic.csv")
