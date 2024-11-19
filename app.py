# Szükséges könyvtárak importálása
import streamlit as st  # Webalkalmazás építésére
import numpy as np  # Számításokhoz és mátrixműveletekhez
import tensorflow as tf  # Mélytanulási modell betöltéséhez és használatához
import cv2  # Képfeldolgozási műveletekhez (pl. binarizálás, átméretezés)
from PIL import Image  # Képformátumok kezelésére
from streamlit_drawable_canvas import st_canvas  # Rajzfelület hozzáadásához

# Modell betöltése és tárolása session state-ben
if 'model' not in st.session_state:
    try:
        # Mentett modell betöltése és újrafordítása
        model = tf.keras.models.load_model('D:/1Sulesz/MestBeadando/emnist_model.keras', compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Modell elérhetővé tétele session state-ben
        st.session_state.model = model
        st.write("MODELL OK")  # Sikeres betöltésről visszajelzés
    except Exception as e:
        st.error(f"Hiba történt a modell betöltésekor: {e}")

# Modell meghívása a session state-ből
model = st.session_state.model

# Karakterlista létrehozása: számjegyek és betűk
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Címsor és utasítások a rajzfelülethez
st.title("Kézírás-felismerő rajzfelület")
st.write("Rajzolj egy számot a rajzfelületre, majd kattints a 'Felismerés' gombra.")

# Rajzfelület létrehozása, ahol a felhasználó rajzolhat
canvas_result = st_canvas(
    fill_color="black",  # Háttérszín
    stroke_width=10,  # Vonalszélesség
    stroke_color="white",  # Vonalszín
    background_color="black",  # Rajzfelület színe
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

# "Felismerés" gomb
if st.button("Felismerés"):
    if canvas_result.image_data is not None:
        # Szürkeárnyalatúvá alakítás és binarizálás (Otsu módszerrel)
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.resize(img_binary, (28, 28))
        img = img.reshape(1, 28, 28, 1) / 255.0  # Normalizálás [0, 1] közé

        # Előrejelzés a modell segítségével
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100  # Bizonyosság (százalékban)

        # Eredmény megjelenítése és oszlopdiagram készítése
        st.write(f"Felismerés eredménye: {characters[predicted_digit]}")
        st.write(f"Biztosság: {confidence:.2f}%")
        st.bar_chart(prediction[0]) 
        
    else:
        st.write("Rajzolj valamit a felismeréshez!")

# Feltöltött képek kezelése és felismerése
st.title("Kép feltöltése")
st.write("Tölts fel egy képet, és a modell megpróbálja felismerni, hogy melyik számjegy látható rajta.")

uploaded_file = st.file_uploader("Válassz egy képfájlt (28x28 px méretű)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Szürkeárnyalatos kép
    img_array = np.array(image)
    st.image(image, caption="Feltöltött kép", width=100)
    # Kép előkészítése felismeréshez (invertálás, átméretezés, normalizálás)
    img_resized = cv2.resize(img_array, (28, 28))
    img_inverted = np.invert(img_resized)
    img_normalized = img_inverted.astype('float32') / 255
    img_normalized = np.expand_dims(img_normalized, axis=(0, -1))

    prediction = model.predict(img_normalized)
    predicted_digit = np.argmax(prediction)
    confidence = prediction[0][predicted_digit] * 100  # Bizonyosság (százalékban)

        # Eredmény megjelenítése és oszlopdiagram készítése
    st.write(f"Felismerés eredménye: {characters[predicted_digit]}")
    st.write(f"Biztosság: {confidence:.2f}%")
    st.bar_chart(prediction[0]) 


st.title("Felismerés webkamerával")

# Webkamera hozzáadása (ha be van kapcsolva)
camera_on = st.checkbox("Webkamera Bekapcsolása")

if camera_on:
    cap = cv2.VideoCapture(0)  # Alapértelmezett kamera megnyitása
    image_placeholder = st.empty()
    chart_placeholder = st.empty()
    result_placeholder = st.empty()
    percent_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Hiba történt a kamera megnyitásakor!")
            break

        # Képkocka feldolgozása: szürkeárnyalatúvá alakítás és binarizálás
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_resized = cv2.resize(img_binary, (28, 28))
        img_resized = img_resized.reshape(1, 28, 28, 1) / 255.0  # Normalizálás

        prediction = model.predict(img_resized)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100  # Bizonyosság (százalékban)


        # Felismerési eredmény frissítése és diagram megjelenítése
        result_placeholder.write(f"Felismerés eredménye: {characters[predicted_digit]}")
        percent_placeholder.write(f"Biztosság: {confidence:.2f}%")
        chart_placeholder.bar_chart(prediction[0])
        image_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Kamera leállítása
    cap.release()
else:
    st.write("A webkamera ki van kapcsolva.")
