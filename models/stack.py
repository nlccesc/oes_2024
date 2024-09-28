from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

class ModelStacker:
    def __init__(self, estimators=None, final_estimator=None):
        if estimators is None:
            estimators = [
                ('xgb', XGBClassifier(random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('knn', KNeighborsClassifier()),
                ('lsvc', SVC(kernel='linear', probability=True, random_state=42)),
            ]
        self.estimators = estimators
        self.final_estimator = final_estimator if final_estimator else RandomForestClassifier(random_state=42)

    def build_stack(self):
        stacking_model = StackingClassifier(estimators=self.estimators, 
                                            final_estimator=self.final_estimator)
        return stacking_model
