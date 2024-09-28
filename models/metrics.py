from sklearn.metrics import f1_score

def evaluate_performance(Y_test_original, predictions):
    f1 = f1_score(Y_test_original, predictions, average='weighted')
    print(f'F1 Score: {f1}')
    
    return f1
