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
warnings.filterwarnings(
    "ignore",
    message="Setting penalty=None will ignore the C and l1_ratio parameters"
)



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
            begin_glm = time.time()
            X_glm = sm.add_constant(X)
            glm_lr = sm.GLM(
                endog = label,
                exog = X_glm,
                family = families.Binomial()
            )
            fitted_glm = glm_lr.fit()
            end_glm = time.time()
            print("\t\t\tTime needed:", round((end_glm - begin_glm)*1e3,2), "(ns)")
            glm_probs = torch.from_numpy(fitted_glm.predict(X_glm))
            glm_probs = torch.stack([1 - glm_probs, glm_probs], dim=1)

        print("\t\t1. scikit learn")
        begin_sklearn = time.time()
        sk_lr = LogisticRegression(C=np.inf, l1_ratio=0, solver="lbfgs")
        sk_lr = sk_lr.fit(X, label)
        end_sklearn = time.time()
        print("\t\t\tTime needed:", round((end_sklearn - begin_sklearn)*1e3,2), "(ns)")

        print("\t\t2. torch implementation")
        begin_torch = time.time()
        torch_lr = TorchLogistic(n_features=X.shape[1], n_classes=label_torch.max()+1)
        torch_lr.fit(X_torch, label_torch, reduction='sum')
        end_torch = time.time()
        print("\t\t\tTime needed:", round((end_torch - begin_torch)*1e3, 2), "(ms)")

        print("\tStep 1: Parameter Comparision. Order (GLM), Sklearn, torch")
        print("\t\tComparision of weights")
        print(torch.cat(
            ([torch.from_numpy(fitted_glm.params[1:]).unsqueeze(0)] if is_binary else []) + [
                torch.from_numpy(sk_lr.coef_), torch_lr.lin_estimator.weight.detach()
                ]
        ))

        print("\n\t\tComparision of intercept")
        print(torch.stack(
            ([torch.from_numpy(fitted_glm.params[[0]])] if is_binary else []) + [
                torch.from_numpy(sk_lr.intercept_), torch_lr.lin_estimator.bias.detach()
            ]
            ))


        print("\tStep 2: Comparision of predicted probabilities.")
        sk_probs = torch.from_numpy(sk_lr.predict_proba(X))
        torch_probs = torch_lr.predict_proba(X_torch).detach()

        # This has to bee completed with the 
        print("\t\tRandomly chosen probs:")
        print(
            torch.stack(
                ([glm_probs] if is_binary else [])+ [sk_probs, torch_probs], 
                dim=1
            )[torch.randint(0, sk_probs.shape[0], size=(6,))]
            )
        print("\t\tAbsolute Distances: (mean and quantiles=.01,.25,.5,.75,.99)")
        mae = lambda a, b : (a-b).abs().mean().detach().item()
        ae_quantiles = lambda a, b, q=torch.tensor([0.01,0.25,0.5,.75,.99]): (a - b).abs().quantile(q)
        print("\t\t\tscikit vs. torch:")
        for f in [mae, ae_quantiles]:
            print("\t\t\t", f(sk_probs, torch_probs))

        if is_binary:            
            print("\t\tGLM vs sklearn:")
            print(mae(glm_probs, sk_probs))
            print(ae_quantiles(glm_probs, sk_probs))

            print("\t\tGLM vs torch:")
            print(mae(glm_probs, torch_probs))
            print(ae_quantiles(glm_probs, torch_probs))
        

