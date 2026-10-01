import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/sample_credit_data.csv")
features=["income","loan_amount","credit_history_years","debt_to_income","late_payments","employment_years"]
sns.boxplot(data=df,x="default",y="debt_to_income"); plt.title("Debt-to-Income Ratio by Outcome"); plt.tight_layout(); plt.savefig(ROOT/"visualisations/dti_by_default.png",dpi=180); plt.close()
X,y=df[features],df["default"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
models={"Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000))]),"Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")}
for name,m in models.items():
    m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; pred=(p>=.5).astype(int)
    print(name,{"ROC-AUC":roc_auc_score(yte,p),"Precision":precision_score(yte,p,zero_division=0),"Recall":recall_score(yte,p,zero_division=0),"F1":f1_score(yte,p,zero_division=0)})
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for name,m in models.items():
    s=cross_validate(m,X,y,cv=cv,scoring=["roc_auc","precision","recall","f1"])
    print(name,{k:s["test_"+k].mean() for k in ["roc_auc","precision","recall","f1"]})
rf=models["Random Forest"]; rf.fit(X,y); pd.Series(rf.feature_importances_,index=features).sort_values().plot(kind="barh",title="Feature Importance"); plt.tight_layout(); plt.savefig(ROOT/"visualisations/feature_importance.png",dpi=180); plt.close()
