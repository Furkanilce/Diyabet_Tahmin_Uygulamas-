import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/diabetes_data_upload.csv")

y = df["class"]
x = df.drop("class", axis = 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)



if os.path.exists("model/xgb_diabets_model.joblib") and os.path.exists("model/encoder.joblib") and os.path.exists("model/lencoder.joblib"):
    model = joblib.load("model/xgb_diabets_model.joblib")
    encoder = joblib.load("model/encoder.joblib")
    lencoder = joblib.load("model/lencoder.joblib")
    print("Model Dosyasından Yüklendi")
else:
    encoder = OneHotEncoder(handle_unknown="ignore")
    lencoder = LabelEncoder()

    encoder.fit(x_train)
    lencoder.fit(y_train)

    x_train_encoded = encoder.transform(x_train)
    y_train_encoded = lencoder.transform(y_train)

    xgb = XGBClassifier()
    model = xgb.fit(x_train_encoded, y_train_encoded)

    joblib.dump(encoder, "encoder.joblib")
    joblib.dump(lencoder, "lencoder.joblib")
    joblib.dump(model, "xgb_diabets_model.joblib")
    print("Model Eğitildi ve Kaydedildi")


x_test_encoded = encoder.transform(x_test)
y_test_encoded = lencoder.transform(y_test)


score = model.score(x_test_encoded, y_test_encoded)
y_pred = model.predict(x_test_encoded)
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)

print("Accuracy = {}".format(accuracy))
print("Precision = {}".format(precision))
print("Recall = {}".format(recall))
print("F1-score = {}".format(f1))
print("Skor = {}".format(score))


st.title("Diyabet Tahmin Uygulaması")

secmeli_sayfa = st.sidebar.selectbox("Sayfa Seç", ["Tahmin Aracı", "Veri Analizi"])

if secmeli_sayfa == "Tahmin Aracı":
    yas = st.number_input("Yaşınızı Giriniz:", min_value = 0, max_value = 100, step = 1)
    cinsiyet = st.radio("Cinsiyet Seçiniz:", ["Kadın", "Erkek"])
    poliüri = st.radio("Poliüri var mı?", ["Evet", "Hayır"])
    polidipsi = st.radio("Polidipsi var mı?", ["Evet", "Hayır"])
    KiloKaybi = st.radio("Ani kilo kaybı yaşıyor musunuz?", ["Evet","Hayır"])
    zayıflık = st.radio("Zayıf mısınız?", ["Evet", "Hayır"])
    polifaji = st.radio("Polifaji var mı?", ["Evet", "Hayır"])
    Genital = st.radio("Genital mantar enfeksionu var mı?", ["Evet", "Hayır"])
    görmeKaybı = st.radio("Görmek kaybı var mı?", ["Evet", "Hayır"])
    kasıntı = st.radio("Kaşıntı var mı?", ["Evet", "Hayır"])
    sinirlilik = st.radio("Sinirlilik var mı?", ["Evet", "Hayır"])
    gecikmisiyilesme = st.radio("Beden iyileşmede gecikem oluyor mu?", ["Evet", "Hayır"])
    kismiParazit = st.radio("Kısmi Parazit var mı?", ["Evet", "Hayır"])
    kasSetligi = st.radio("Kas sertliği var mı?", ["Evet", "Hayır"])
    alopesi = st.radio("Alopsei var mı?", ["Evet", "Hayır"])
    obezite = st.radio("Obazite var mı?", ["Evet", "Hayır"])


    if st.button("Gönder"):
        data = []
        if yas >= 0 and yas <= 100:
            data.append(yas)
        else:
            st.write("Doğru bir yaş giriniz.")

        if cinsiyet == "Kadın":
            data.append("Female")
        else:
            data.append("Male")

        if poliüri == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if polidipsi == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if KiloKaybi == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if zayıflık == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if polifaji == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if Genital == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if görmeKaybı == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if kasıntı == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if sinirlilik == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if gecikmisiyilesme == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if kismiParazit == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if kasSetligi == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if alopesi == "Evet":
            data.append("Yes")
        else:
            data.append("No")

        if obezite == "Evet":
            data.append("Yes")
        else:
            data.append("No")


        data_df = pd.DataFrame([data], columns = x.columns)
        data_encoded = encoder.transform(data_df)
        prediction = model.predict(data_encoded)
        prediction_str = lencoder.inverse_transform(prediction)

        probs = model.predict_proba(data_encoded)
        max_proba = np.max(probs)

        st.write("Sonuç:")
        if prediction_str == "Positive":
            st.error(f"Diyabet riskiniz **yüksek**. (%{max_proba * 100:.2f} olasılık)\n\nBir doktora görünmenizi tavsiye ederiz.")

        else:
            st.success(f"Diyabet riskiniz **düşük**. (%{max_proba * 100:.2f} olasılık)")

elif secmeli_sayfa == "Veri Analizi":
    st.subheader("📊 Sınıf ve Cinsiyet Dağılımı")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="class", hue="Gender", ax=ax)
    ax.set_title("Diyabet Durumunun Cinsiyete Göre Dağılımı")
    st.pyplot(fig)

