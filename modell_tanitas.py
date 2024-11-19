import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


#EMNIST CSV adatok betöltése
data = pd.read_csv('D:/1Sulesz/MestBeadando/eminst/emnist-byclass-train.csv')  # útvonalat megadni 

# Címkék és képadatok kinyerése
y = data.iloc[:, 0].values  # Az első oszlop a címke
X = data.iloc[:, 1:].values  # Az összes többi oszlop a képadat

# Adatok előkészítése
X = X.reshape(-1, 28, 28, 1)
X = X / 255.0  # Normalizálás 0-1 közötti tartományba

# Adatok felosztása tréning és teszt adathalmazokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Konvolúciós neurális hálózat építése
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(62, activation='softmax')  # 62 kimeneti osztály (betűk és számok)
])

#Modell fordítása
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Modell betanítása
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

#Modell kiértékelése
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Teszt pontosság: {test_acc:.2f}')

#Modell mentése
model.save('emnist_model.keras')
print("A modell elmentve emnist_model néven.")

