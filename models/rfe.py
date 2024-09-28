from sklearn.feature_selection import RFE
import numpy as np
from sklearn.model_selection import cross_val_score
from copy import deepcopy

class FeatureSelector:
    def __init__(self, model):
        self.model = model

    def en_rfe(self, X, Y, num_features):
        """
        Enhanced Recursive Feature Elimination (EnRFE) implementation.
        """
        n_features = X.shape[1]
        remaining_features = list(range(n_features))
        selected_features = []
        best_score = -np.inf

        while remaining_features:
            best_feature_to_remove = None

            for feature in remaining_features:
                current_features = deepcopy(remaining_features)
                current_features.remove(feature)

                # Train the model on the remaining features
                self.model.fit(X[:, current_features], Y)
                score = cross_val_score(self.model, X[:, current_features], Y, cv=5).mean()

                if score > best_score:
                    best_score = score
                    best_feature_to_remove = feature

            if best_feature_to_remove is not None:
                remaining_features.remove(best_feature_to_remove)
                selected_features.append(best_feature_to_remove)

            # Stop if we have selected the desired number of features
            if len(remaining_features) == num_features:
                break

        # Final selected features
        final_selected_features = sorted(selected_features)
        print(f"Selected features by EnRFE: {final_selected_features}")

        return X[:, final_selected_features], final_selected_features

    def select_features(self, X, Y, num_features):
        """
        Combine standard RFE with the Enhanced RFE (EnRFE) approach.
        """
        # First step: Perform standard RFE
        rfe = RFE(estimator=self.model, n_features_to_select=num_features)
        X_rfe = rfe.fit_transform(X, Y)
        selected_features_rfe = rfe.support_

        print(f"Selected features by RFE: {selected_features_rfe}")

        # Second step: Use feature importance from the model
        self.model.fit(X, Y)
        feature_importances = self.model.feature_importances_

        # Selecting top features based on importance
        indices = np.argsort(feature_importances)[::-1]
        top_indices = indices[:num_features]
        selected_features_importance = np.zeros_like(selected_features_rfe, dtype=bool)
        selected_features_importance[top_indices] = True

        print(f"Top {num_features} features by importance: {top_indices}")

        # Combining RFE and importance-selected features (intersection)
        combined_features = selected_features_rfe & selected_features_importance
        X_combined = X[:, combined_features]

        # Third step: Apply Enhanced RFE on the combined features
        X_enrfe, final_selected_features = self.en_rfe(X_combined, Y, num_features)

        return X_enrfe, final_selected_features
