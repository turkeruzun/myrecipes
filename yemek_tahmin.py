import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Geniş modda başlatmak için set_page_config kullanın
st.set_page_config(layout="wide",page_icon="🧊",page_title="Yemek Tarifleri Zorluk Tahmini ve Öneri Sistemi")
# <img src="https://www.cdnlogo.com/logos/m/2/my-recipes.svg" alt="Logo" style="width: 180px; margin-bottom: 20px;">
def main():
    # Header ve tablar için gerekli HTML kodu
    st.markdown(
    """
    <div style="text-align:center">
        <img src="https://www.cdnlogo.com/logos/m/2/my-recipes.svg" alt="Logo" style="width: 250px; margin-bottom: 10px;">
    </div>
    """,unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["TARİF ZORLUK TAHMİNİ", "TARİF ÖNERİ SİSTEMİ", "Algorizm aka FREGX Kim?"])
    with tab1:
        st.header("TARİF ZORLUK TAHMİNİ")
        tab1_zorluk_tahmin()
    with tab2:
        st.header("TARİF ÖNERİ SİSTEMİ")
        # Veri yükleme - Sadece bir kere yüklenecek
        @st.cache_resource
        def load_data():
            return pd.read_parquet("kucukveri.parquet")
        # Veri yükleme
        df = load_data()
        # Cosine similarity için CountVectorizer oluşturma
        vectorizer = CountVectorizer()
        # Kullanıcıdan yemek kategorisi ve zorluk derecesi seçimini alma
        filtre_kolon=['Balik', 'Kirmizi_ET', 'Beyaz_ET', 'Saglikli', 'Vegan', 'Bebek', 'Cocuk', 'Meyveli', "Alkol", "Hamur_isi", "Tatli"]
        yemek_kategorisi = st.selectbox("Yemek Kategorisi Seçin", filtre_kolon)
        zorluk_derecesi = st.selectbox("Zorluk Derecesi Seçin", ["Kolay", "Zor"])

        # "Tarifi Getir" butonu - Kategori ve zorluk seçimi tamamlanmadan çalışmayacak
        if yemek_kategorisi and zorluk_derecesi:
            filtered_recipes = df[(df[yemek_kategorisi] == 1) & (df['Zorluk_Seviye'] == zorluk_derecesi)]
            if st.button("Rasgele Bir Tarif Getir"):
                global random_recipe
                if not filtered_recipes.empty:
                    random_recipe = filtered_recipes.sample(n=1)
                    st.subheader("Önerilen Tarif:")
                    st.write(f"""
                            **Yemek Adı:** {random_recipe['Yemek_Adi'].values[0].capitalize()} \n
                            **Malzemeler:** {random_recipe['NER'].values[0]} \n
                            **Tarif:** {random_recipe['Tarif'].values[0]} \n
                            **Tahmini DK:** {random_recipe['Toplam_Tarif_DK'].values[0]} \n
                            {'**Fırın Derecesi:** ' + str(random_recipe['Fırın_Sıcaklığı'].values[0]) + " F" if random_recipe['Fırın_Yemeği_Mi'].values[0] == 1 else ''} \n
                            **Index:** {random_recipe['Index'].values[0]} """)
                else:
                    st.warning("Seçilen kategori ve zorluk derecesinde tarif bulunamadı.")

            if st.button("Benzerleriyle Birlikte Tarif Getir"):
                random_recipe = filtered_recipes.sample(n=1)
                if 'random_recipe' in globals() and not random_recipe.empty:
                    selected_recipe_name = random_recipe['Yemek_Adi'].values[0]
                    selected_recipe_ingredients = random_recipe['NER'].values[0]
                    selected_recipe_detail = random_recipe['Tarif'].values[0]
                    selected_recipe_index = random_recipe['Index'].values[0]
                    selected_recipe_zorluk = random_recipe['Zorluk_Seviye'].values[0]

                    selected_recipe_text = f"{selected_recipe_name} {' '.join(selected_recipe_ingredients)}"
                    all_recipes_texts = [f"{name} {' '.join(ingredients)}" for name, ingredients in zip(df['Yemek_Adi'], df['NER'])]

                    X = vectorizer.fit_transform([selected_recipe_text] + all_recipes_texts)
                    cosine_similarities = cosine_similarity(X[0:1], X[1:]).flatten()
                    most_similar_recipe_indices = cosine_similarities.argsort()[-4:-1][::-1]
                    most_similar_cosine_values = cosine_similarities[most_similar_recipe_indices]


                    st.subheader("Önerilen Tarif:")
                    st.write("**Yemek Adı:**", selected_recipe_name)
                    st.write("**Malzemeler:**", selected_recipe_ingredients)
                    st.write("**Tarif Detay:**", selected_recipe_detail)
                    st.write("**Index:**", selected_recipe_index)
                    st.write("**Zorluk Seviye:**", selected_recipe_zorluk)


                    st.subheader("Benzer Tarifler:")
                    for idx, cosine_value in zip(most_similar_recipe_indices, most_similar_cosine_values):
                        st.write(f"""
                            **Yemek Adı:** {df.iloc[idx]['Yemek_Adi']} \n
                            **Benzerlik Oranı:** %{round(cosine_value*100,2)} \n

                            **Malzemeler:** {df.iloc[idx]['NER']} \n

                            **Tarif:** {df.iloc[idx]['Tarif']} \n

                            **Tahmini DK:** {df.iloc[idx]['Toplam_Tarif_DK']} \n
                            {'**Fırın Derecesi:** ' + str(df.iloc[idx]['Fırın_Sıcaklığı']) if df.iloc[idx]['Fırın_Yemeği_Mi'] == 1 else ''} \n
                            **Index:** {int(df.iloc[idx]['Index'])} \n
                            **Zorluk Seviyesi:** {df.iloc[idx]['Zorluk_Seviye']} \n

                            {'-' * 60} 
                        """)
                else:
                    st.warning("Önce bir tarif getirin.")

    with tab3:
    #     # Teams renk şeması
    #     primary_color = "#0078D4"
    #     secondary_color = "#FFFFFF"
    #     accent_color = "#F3F2F1"
        st.subheader("Algorizm aka FREGX Kim?")
    #     # Tanıtım içeriği
    #     st.write("""
    #     Algorizm aka FREGX, bu uygulamanın arkasındaki zeki algoritma ve yapay zeka sistemidir. 
    #     FREGX, kullanıcıların tariflerini analiz ederek zorluk seviyesini tahmin eder ve kullanıcılara uygun öneriler sunar.
    #     Bu sistem sayesinde kullanıcılar daha kolay ve keyifli bir yemek deneyimi yaşayabilirler.

    #     FREGX'in geliştirilmesi, lezzetli yemek tariflerini keşfetmek isteyen herkes için bir rehber olmayı amaçlamaktadır.
    #     """)

    #     # Resimler ve bilgiler
    #     st.markdown(
    #         """
    #         <div style="padding:20px;border-radius:10px;text-align:center;">
    #             <div style="display:flex; justify-content: space-between;">
    #                 <div style="flex: 0 0 20%; text-align: center;">
    #                     <img src="https://image.freepik.com/free-vector/data-analysis-concept-illustration_114360-1060.jpg" alt="FREGX" style="width: 100%; border-radius:10px;">
    #                     <p style="color:white;"><h4>Türker UZUN</h4></p>
    #                     <p style="color:white;">Veri Bilimci</p>
    #                 </div>
    #                 <div style="flex: 0 0 20%; text-align: center;">
    #                     <img src="https://image.freepik.com/free-vector/data-analysis-concept-illustration_114360-1060.jpg" alt="FREGX" style="width: 100%; border-radius:10px;">
    #                     <p style="color:white;"><h4>İlhami DEMİRCİ</h4></p>
    #                     <p style="color:white;">Veri Analisti</p>
    #                 </div>
    #                 <div style="flex: 0 0 20%; text-align: center;">
    #                     <img src="https://image.freepik.com/free-vector/data-analysis-concept-illustration_114360-1060.jpg" alt="FREGX" style="width: 100%; border-radius:10px;">
    #                     <p style="color:white;"><h4>Vildan ÇİMEN</h4></p>
    #                     <p style="color:white;">Makine Öğrenimi Mühendisi</p>
    #                 </div>
    #                 <div style="flex: 0 0 20%; text-align: center;">
    #                     <img src="https://image.freepik.com/free-vector/data-analysis-concept-illustration_114360-1060.jpg" alt="FREGX" style="width: 100%; border-radius:10px;">
    #                     <p style="color:white;"><h4>Mahmut KEÇECİ</h4></p>
    #                     <p style="color:white;">Veri Mimarısı</p>
    #                 </div>
    #             </div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     # Footer
    #     st.markdown(
    #         f"""
    #         <hr>
    #         <p style="color:{primary_color};text-align:center;">© 2024 Algorizm - Yemek Tarifi Zorluk Tahmin ve Öneri Sistemi</p>
    #         """,
    #         unsafe_allow_html=True
    #     )

def tab1_zorluk_tahmin():
    # CSS ile sütunlar arasına padding eklemek için stil
    st.markdown("""
        <style> .padding-between-columns > div {
                padding: 0 70px; /* İstenilen padding değerini buradan ayarlayabilirsiniz */
                box-sizing: border-box; } </style>
        """, unsafe_allow_html=True)
    
    # Sütunları oluştur
    col1, col2, col3 = st.columns([1.5, 0.25, 1.25])

    # Sol sütun (form)
    with col1:
        # Soruları sormak için kullanıcı girişi al
        malzeme_cesit_sayisi = st.number_input("1. Soru: Tarifinizde kaç çeşit malzeme kullanıyorsunuz?", value=0)
        tarif_adim_sayisi = st.number_input("2. Soru: Tarifiniz kaç adımda gerçekleşiyor?", value=0)
        toplam_gramaj = st.number_input("3. Soru: Tarifiniz kaç gram?", value=0)
        tarif_metni = st.text_area("4. Soru: Tarifin metnini yapıştırın")
        kategori = st.selectbox("5. Soru: Tarifiniz hangi kategoride?", options=["Baslangic","Ana_Yemek", "Sebze", "Tatli", "Saglikli"])
        sicak_islem = st.radio("6. Soru: Sıcak İşlem Var Mı?", options=["Evet", "Hayır"])
        firin_islem = st.radio("7. Soru: Fırın İşlemi Var Mı?", options=["Evet", "Hayır"])
        alkollu_tarif = st.radio("8. Soru: Tarifte alkol kullanılıyor mu?", options=["Evet", "Hayır"], index=1)

        # Butona basıldığında işlemi gerçekleştir
        if st.button("Zorluk Seviyesini Hesapla"):
            # Kullanıcının girdiği değerleri işle
            if malzeme_cesit_sayisi != 0 and tarif_adim_sayisi != 0 and toplam_gramaj != 0 and tarif_metni != "" and kategori != "" and sicak_islem != "":
                # Sıcak yemeği işle
                sicak = 1 if sicak_islem == "Evet" else 0
                firin = 1 if firin_islem == "Evet" else 0
                alkol = 1 if alkollu_tarif == "Evet" else 0

                # Ana yemeği işle
                kategoriler = {"Başlangıç":1,  "Sebze": 2, "Ana_Yemek": 3, "Tatli":4, "Saglikli":5}
                kategori = kategoriler.get(kategori, 0)

                # Tatlı, çorba, salata veya sebze seçilmişse diğerlerini sıfırla
                if kategori != 1:
                    tatli, corba, salata, sebze = (0, 0, 0, 0)

                # Yeni tarif verisini oluştur
                new_recipe = {
                            'Malzeme_Cesit_Sayisi': malzeme_cesit_sayisi,
                            'Tarif_Adim_Sayisi': tarif_adim_sayisi,
                            'Tarif_Char': len(tarif_metni),
                            'Toplam_Gramaj': toplam_gramaj,
                            "Baslangic":0,
                            "Ana_Yemek":1,
                            'Sebze':0,
                            'Saglikli':0,
                            'Tatli':0,
                            'Alkol':alkol,
                            'Sicak':sicak,
                            'Fırın_Yemeği_Mi':firin
                        }
                
                import joblib
                ensemble_model = joblib.load("ensemble_model_st.pkl")

                # Yeni tarifi DataFrame formatına dönüştür
                new_recipe_df = pd.DataFrame(new_recipe, columns=ensemble_model.feature_names_in_, index=[0])

                # Modeli kullanarak tahmin yap
                new_recipe_difficulty_pred = ensemble_model.predict(new_recipe_df)

                # Tahmin sonucunu gösterme
                if new_recipe_difficulty_pred[0] == 0:
                    st.success("Yeni tarifin zorluk tahmini: Kolay")
                else:
                    st.error("Yeni tarifin zorluk tahmini: Zor")
            else:
                st.warning("Lütfen tüm soruları cevaplayın.")
    # Sağ sütun (görsel)
    # CSS ile animasyon tanımlama
    st.markdown("""
        <style>
            @keyframes rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            .rotating-image {
                animation: rotate 12s linear infinite;
            }
        </style>
    """, unsafe_allow_html=True)

    with col3:
        st.markdown('<img src="https://themewagon.github.io/restoran/img/hero.png" class="rotating-image" width="580">', unsafe_allow_html=True)
        # st.image("img/yemek_sag.png", use_column_width=False, width=600)


if __name__ == "__main__":
    main()
