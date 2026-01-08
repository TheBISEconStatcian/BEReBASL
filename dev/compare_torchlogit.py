# ============================
# Standard library imports
# ============================
import time
import warnings

# ============================
# Third-party imports
# ============================
import numpy as np
import torch
import statsmodels.api as sm
from statsmodels.genmod import families
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# ============================
# Local project imports
# ============================
from berebasl.BASL.classifiers import TorchLogistic

# ============================
# Global configuration
# ============================
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=6)
# Supurious warning which was corrected in github repo but not
# in the current sci-kit learn release
warnings.filterwarnings(
    "ignore",
    message="Setting penalty=None will ignore the C and l1_ratio parameters"
)

def fit_glm(X : np.array, label : np.array):
    begin_glm = time.time()
    X_glm = sm.add_constant(X)
    glm_lr = sm.GLM(
        endog = label,
        exog = X_glm,
        family = families.Binomial()
    )
    fitted_glm = glm_lr.fit()
    end_glm = time.time()

    train_time = end_glm - begin_glm

    begin_inference = time.time()
    glm_probs = fitted_glm.predict(X_glm)
    glm_probs = np.column_stack([1 - glm_probs, glm_probs])
    end_inference = time.time()
    inference_time = end_inference - begin_inference

    return (fitted_glm, X_glm), glm_probs, train_time, inference_time

def fit_sklearn(X, label : np.array):
    begin_sklearn = time.time()
    sk_lr = LogisticRegression(C=np.inf, l1_ratio=0, solver="lbfgs")
    sk_lr = sk_lr.fit(X, label)
    end_sklearn = time.time()

    train_time = end_sklearn - begin_sklearn

    begin_inference = time.time()
    sk_probs = sk_lr.predict_proba(X)
    end_inference = time.time()
    inference_time = end_inference - begin_inference

    return sk_lr, sk_probs, train_time, inference_time

def fit_torch(X_torch, label_torch):
    begin_torch = time.time()
    torch_lr = TorchLogistic(n_features=X.shape[1], n_classes=label_torch.max()+1)
    torch_lr.fit(X_torch, label_torch, reduction='sum')
    end_torch = time.time()

    train_time = end_torch - begin_torch

    begin_inference = time.time()
    torch_probs = torch_lr.predict_proba(X_torch)
    end_inference = time.time()
    inference_time = end_inference - begin_inference

    return torch_lr, torch_probs.detach(), train_time, inference_time

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data          # shape (150, 4)
    X_torch = torch.from_numpy(X)
    y = iris.target.astype(float)        # shape (150,)
    feature_names = iris.feature_names
    target_names = iris.target_names

    wished_target = "versicolor"


    y_bin = (y == np.where(target_names==wished_target)[0].item()).astype(float)

    for label, desc in [(y_bin, "binary"), (y, "multiclass")]:
        is_binary = desc == "binary"
        label_torch = torch.from_numpy(label)
        label_torch = label_torch.unsqueeze(-1) if is_binary else label_torch.to(int)
        print("\nBeginning", desc, "comparision:")

        print("\tStep 0: Estimation")
        if is_binary: # This gives an error
            print("\t\t0. GLM:")
            (fitted_glm, X_glm), glm_probs, train_time, _ = fit_glm(X, label)
            
            print("\t\t\tTime needed:", round(train_time*1e3,2), "(ms)")

            glm_probs = torch.from_numpy(glm_probs)
            glm_probs = torch.stack([1 - glm_probs, glm_probs], dim=1)

        print("\t\t1. scikit learn")
        begin_sklearn = time.time()
        sk_lr = LogisticRegression(C=np.inf, l1_ratio=0, solver="lbfgs")
        sk_lr = sk_lr.fit(X, label)
        end_sklearn = time.time()
        print("\t\t\tTime needed:", round((end_sklearn - begin_sklearn)*1e3,2), "(ms)")

        print("\t\t2. torch implementation")
        begin_torch = time.time()
        torch_lr = TorchLogistic(n_features=X.shape[1], n_classes=label_torch.max()+1)
        torch_lr.fit(X_torch, label_torch, reduction='sum')
        end_torch = time.time()
        print("\t\t\tTime needed:", round((end_torch - begin_torch)*1e3, 2), "(ms)")

        print("\tStep 1: Parameter Comparision. Order (GLM), Sklearn, torch")
        print("\t\tComparision of weights")
        if is_binary:
            joint_weights = torch.cat([
                torch.from_numpy(fitted_glm.params[1:]).unsqueeze(0), 
                torch.from_numpy(sk_lr.coef_), 
                torch_lr.lin_estimator.weight.detach()
            ])
        else:
            joint_weights = torch.stack([
                torch.from_numpy(sk_lr.coef_), 
                torch_lr.lin_estimator.weight.detach()
            ], dim = 1)
        print(joint_weights)

        print("\n\t\tComparision of intercept")
        print(torch.stack(
            ([torch.from_numpy(fitted_glm.params[[0]])] if is_binary else []) + [
                torch.from_numpy(sk_lr.intercept_), torch_lr.lin_estimator.bias.detach()
            ]
            ))


        print("\tStep 2: Comparision of predicted probabilities.")
        sk_probs = torch.from_numpy(sk_lr.predict_proba(X))
        torch_probs = torch_lr.predict_proba(X_torch).detach()
        joint_probs = torch.stack(
                ([glm_probs] if is_binary else [])+ [sk_probs, torch_probs], 
                dim=1
            )

        print("\t\tRandomly chosen probs:")
        print(joint_probs[torch.randint(0, sk_probs.shape[0], size=(6,))])
        print("\t\tAbsolute Distances: (mean and quantiles=.01,.25,.5,.75,.99)")
        mae = lambda a, b : (a-b).abs().mean().detach().item()
        ae_quantiles = lambda a, b, q=torch.tensor([0.01,0.25,0.5,.75,.99]): (a - b).abs().quantile(q)

        comp_spec = [
            ((sk_probs, torch_probs), "scikit vs. torch:")
        ]
        if is_binary:
            comp_spec += [
                ((sk_probs, glm_probs), "scikit vs. GLM:"),
                ((glm_probs, torch_probs), "GLM vs. torch:"),
                ((glm_probs, sk_probs), "GLM vs. scikit:"),
            ]

        for pair, desc_pair in comp_spec:
            print('\t'*3, desc_pair)
            for f in [mae, ae_quantiles]:
                print('\t'*4, f(*pair))
        
        print("\tStep 3: Comparision of predictions (argmax threshold)")
        preds = joint_probs.argmax(dim=-1, keepdim=False)
        predicts_the_same_class = (preds == preds[:, :1]).all(dim=1)
        print("\t\tProportion equal predictions among models:", 
              (predicts_the_same_class.sum() / predicts_the_same_class.shape[0]).round(decimals=2).item() * 100,
              "%")

