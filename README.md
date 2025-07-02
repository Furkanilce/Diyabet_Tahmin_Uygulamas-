Bu proje, XGBoost algoritması kullanarak kullanıcıdan alınan tıbbi ve kişisel bilgileri değerlendirip diyabet riskini tahmin eden bir Streamlit web uygulamasıdır.

   ## 📚 İlham Kaynağı / Referans
  
  Bu proje, IEEE Xplore’da yayımlanmış olan  
  **“Use of Machine Learning Techniques to Predict Diabetes at an Early Stage”**  
  adlı akademik çalışmadan esinlenilerek hazırlanmıştır.
  
  📄 **Makale Bilgisi**:  
  M. S. Islam, M. A. Rahman ve M. R. Islam,  
  *Use of Machine Learning Techniques to Predict Diabetes at an Early Stage*,  
  2021 International Conference on Networking and Advanced Systems (ICNAS), Annaba, Algeria,  
  27-28 October 2021, 
  
  🔗 [IEEE Xplore'da Görüntüle](https://ieeexplore.ieee.org/document/9628903)  
  
  📌 DOI: [10.1109/ICNAS53565.2021.9628903](https://doi.org/10.1109/ICNAS53565.2021.9628903)

  ## 📊 Veri Seti Hakkında

Bu projede kullanılan veri seti,  
**“Use of Machine Learning Techniques to Predict Diabetes at an Early Stage”**  
adlı IEEE Xplore’da yayımlanmış akademik çalışmaya ait veriler kullanılarak oluşturulmuştur.  
Veri seti, makalenin yazarları tarafından toplanmış ve işlenmiştir.

[Veri Seti Linki](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)

Özellikler:

 - Kullanıcı dostu arayüz ile kolay veri girişi
 - Diyabet riskini % olasılıkla tahmin etme
 - Model eğitimi ve kodlama için Python, scikit-learn, XGBoost ve Streamlit kullanıldı
 - Model ve encoder dosyaları joblib ile saklanmakta ve yüklenmektedir
 - Tahmin sonuçları kullanıcıya anlaşılır şekilde sunulmaktadır

Kullanılan Teknolojiler ve Kütüphaneler:

 - Python 3.x
 - pandas
 - numpy
 - scikit-learn
 - xgboost
 - joblib
 - streamlit

Kullanım:

  Projeyi klonlayın veya indirin
  
  Gerekli kütüphaneleri yükleyin:
  ```bash
    pip install -r requirements.txt
   ``` 
  Uygulamayı çalıştırın:
  ```bash
    streamlit run app.py
  ```
Tarayıcıda açılan arayüzden bilgilerinizi girerek diyabet riskinizi öğrenin.

Lisans:
  Bu proje MIT lisansı ile lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakabilirsiniz.


