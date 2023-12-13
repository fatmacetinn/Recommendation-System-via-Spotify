from typing import List, Any
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df= pd.read_csv(r'D:\SpotifyVeri\spotify veriler.csv')
df.head(200)

df.columns
df["track_name"].head(100)
df["album_name"].head()
df.info()
df["track_genre"].head()
df["artists"].head()


################################################
#TEMİZLEME ADIMLARI

#eksik değer var-yok
df.isnull().values.sum()
#değişkenlerdeki tam değer sayısı
df.notnull().sum()
#veri setindekitoplam eksik değer sayısı
df.isnull().sum().sum()
#en az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]
#tam olan gözlem birimleri
df[df.notnull().all(axis=1)]
#oransal olarak görmek için
(df.isnull().sum()/df.shape[0]*100).sort_values(ascending=False)
df.drop('Unnamed: 0', axis=1, inplace=True)

#########################################################################################################

#KORELASYON ANALİZİ

df.plot.scatter(x="energy", y="loudness", color="blue")
plt.show()
df["energy"].corr(df["loudness"])


df.plot.scatter(x="energy", y="tempo", color="blue")
plt.show()
df["tempo"].corr(df["energy"])

#korelasyon
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
corr = df[num_cols].corr()
print(corr)
sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr,cmap="RdBu")
plt.show()


################################################
#fonksiyon olmayan kod
df.fillna('', inplace=True)
ozellikler = ['artists', 'album_name', 'track_genre']
# Özellik vektörlerini oluşturdum vektorizer kullanarak
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df[ozellikler].astype(str).apply(lambda x: ' '.join(x), axis=1))

# Benim seçtiğim şarkıya benzer öneriler yapmasını isteyeceğim
secilen_track_name = "Hold On"

# Seçilen şarkının indeksini bulup o index üzerinden arama yapacak
secilen_index = df[df['track_name'] == secilen_track_name].index[0]

# Seçtiğim şarkının özellik vektörünü oluşturdum
secilen_vector = feature_vectors[secilen_index]

# Benzerlikleri
similarities = cosine_similarity(secilen_vector.reshape(1, -1), feature_vectors)

num_recommendations = 10
top_indices = similarities.argsort()[0][::-1][:num_recommendations]  # Seçilen şarkıyı hariç tuttum
# Önerilen şarkılar
onerilen_sarki = df.iloc[top_indices]
print(onerilen_sarki)


def oneri_similar_sarki(df, secilen_track_name, oneri_sarki_sayisi=10):

    features = ['artists', 'album_name', 'track_genre']

    df.fillna('', inplace=True)

    # Özellik vektörlerini oluşturun
    vectorizer = TfidfVectorizer()
    ozellik_vectors = vectorizer.fit_transform(df[features].astype(str).apply(lambda x: ' '.join(x), axis=1))

    secilen_index = df[df['track_name'] == secilen_track_name].index[0]

    secilen_vektor = ozellik_vectors[secilen_index]

    # Benzerlikleri hesapladım
    similarities = cosine_similarity( secilen_vektor .reshape(1, -1),ozellik_vectors)

    # Benzer şarkıların en iyi indekslerini
    top_indices = similarities.argsort()[0][-oneri_sarki_sayisi-1:-1]  # Seçilen şarkıyı hariç tutun

    onerilen_sarki = df.iloc[top_indices]

    filtrele_oneri_sarkilari =  onerilen_sarki[onerilen_sarki['track_name'] !=secilen_track_name]

    filtrele_oneri_sarkilari  =  filtrele_oneri_sarkilari[['album_name', 'track_name']]

    return filtrele_oneri_sarkilari .head(oneri_sarki_sayisi)

oneri_similar_sarki(df,"Hold On",5)



########################################
#FARKLI ÖZELLİKLERE GÖRE SIRALAYALIM
ozellikler2 = ['danceability', 'energy', 'popularity']

df[ozellikler2] = df[ozellikler2].fillna(df[ozellikler2].mean())

secilen_track_name = "Hold On"

# Seçilen şarkının özellikleri
selected_track = df[df['track_name'] == secilen_track_name ][ozellikler2].values[0]

# Öneri listesi
onerilenler = df[ozellikler2].apply(lambda x: pd.Series({'similarity': cosine_similarity([selected_track], [x.values])[0][0]}), axis=1)
onerilenler = pd.concat([df['track_name'],onerilenler], axis=1)

# Seçilen şarkıyı öneri listesinden çıkar
onerilenler = onerilenler[onerilenler['track_name'] != secilen_track_name ]

# Öneri listesini sırala
sorted_recommendations =onerilenler.sort_values(by='similarity', ascending=False)

top_10_onerilenler = sorted_recommendations['track_name'].head(10)

print(top_10_onerilenler)

#FARKLI TÜR ŞARKILARIN SEÇİLEN TÜR İSİMLERİNE GÖRE POPÜLER OLANLARIN ÖNERİLMESİ

#Tür isimlerini öğrenelim
unique_tur = df['track_genre'].unique()
say_unique_tur= len(unique_tur)
print("Toplam farklı tür sayısı:", say_unique_tur)
print("Farklı türlerin isimleri:")
for tur in unique_tur:
    print(tur)

#Seçtiğim türe göre öneri sıralaması yapalım

 secilen_tur = input("Lütfen bir tür ismi girin: ")
 filtrele_df = df[df['track_genre'] == secilen_tur]
 sorted_df = filtrele_df .sort_values(by='popularity', ascending=False)
 top_sarki = sorted_df[['track_name', 'album_name', 'popularity', 'artists']].head(10)
 print("Seçilen Tür:", secilen_tur)
 print(top_sarki)

