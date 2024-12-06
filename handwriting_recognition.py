#generate dataset

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


digits = datasets.load_digits()

x = digits.images
y = digits.target

n_samples = len(x)
#flattening the image from 2D to 1D
x = x.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=12, shuffle = True)


#fit the model and calculate accuracy
svc_model = SVC(gamma=0.001, C = 0.1)


svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix of SVC Model")
plt.show()


#prediction in paint

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = (img_array/16).astype(np.float64)
    

    return img_array.flatten()

test_image_path = 'test_digit.png'


test_image = preprocess_image(test_image_path)

plt.imshow(test_image.reshape(8, 8), cmap='gray')
plt.title("Preprocessed Custom Image")
plt.colorbar()
plt.show()


prediction = svc_model.predict([test_image])
print(f"Predicted digit from custom image: {prediction[0]}")





