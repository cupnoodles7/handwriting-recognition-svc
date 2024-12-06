from sklearn.datasets import load_digits
import pandas as pd

# Load digits dataset
digits = load_digits()

# Create DataFrame
data = pd.DataFrame(data=digits.data, columns=digits.feature_names)
data['target'] = digits.target
