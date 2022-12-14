import streamlit as st
import twint
import pandas as pd
from functions import *
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Set page name and favicon
st.set_page_config(page_title='Clustering Data',page_icon=':iphone:')
st.image('data:image/jpeg;base64,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')
st.subheader("""
Mari melakukan crawling data dengan menyenagkan!!!:
""")

# customize form
# with st.form(key='Twitter_form'):
#     search_term = st.text_input('Input data yang dicari')
#     limit = st.slider('Banyak Abstrak yg diinginkan', 0, 500, step=20)
#     output_csv = st.radio('Simpan file CSV?', ['Ya', 'Tdk'])
#     file_name = st.text_input('Nama file CSV:')
#     submit_button = st.form_submit_button(label='Cari')

#     if submit_button:
#         # configure twint
#         c = twint.Config()
#         c.Search = search_term
#         c.Limit = limit
#         c.Store_csv = True

#         if c.Store_csv:
#             c.Output = f'{file_name}.csv'

#         twint.run.Search(c)

#         data = pd.read_csv(f'{file_name}.csv', usecols=['date', 'Abstrak'])
#         st.table(data)

# try:
#     st.download_button(label='Download results', data=convert_df(data), file_name = f'{file_name}.csv', mime='text/csv')
# except:
#     pass

st.subheader("Preprocessing Data")
st.markdown("""
proses untuk menyeleksi data text agar menjadi lebih terstruktur lagi dengan 
melalui serangkaian tahapan yang meliputi tahapan case folding, tokenizing, filtering dan stemming
""")
# uploaded_file = st.file_uploader("upload file")
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write(type(data))
#     data = uploaded_file
#     st.write(type(data))

data = pd.read_csv("https://drive.google.com/drive/u/0/folders/10PHejy-KrwiXiCnfcVtPr7hAIMPRKEUF")
st.write(data)
data['Abstrak'] = data['Abstrak'].str.lower()
st.dataframe(data)

def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")                
data['Abstrak'] = data['Abstrak'].apply(remove_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)
data['Abstrak'] = data['Abstrak'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))
data['Abstrak'] = data['Abstrak'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()
data['Abstrak'] = data['Abstrak'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)
data['Abstrak'] = data['Abstrak'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)
data['Abstrak'] = data['Abstrak'].apply(remove_singl_char)

# token
nltk.download('punkt')
# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)
data['Abstrak'] = data['Abstrak'].apply(word_tokenize_wrapper)

# filtering
nltk.download('stopwords')
list_stopwords = stopwords.words('indonesian')
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])
# convert list to dictionary
list_stopwords = set(list_stopwords)
#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]
data['Abstrak'] = data['Abstrak'].apply(stopwords_removal)

# stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)
term_dict = {}

for document in data['Abstrak']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")

# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]
data['Abstrak'] = data['Abstrak'].swifter.apply(get_stemmed_term)
data['Abstrak'].to_csv('Prepocessing.csv',index=False)
# st.table(data['Abstrak'])
st.dataframe(data['Abstrak'])

st.subheader("TF-IDF")
st.markdown("""
Algoritma TF-IDF (Term Frequency – Inverse Document Frequency) adalah salah satu algoritma yang dapat digunakan untuk menganalisa hubungan antara sebuah frase/kalimat dengan sekumpulan dokumen
""")
st.latex("$$tf-idf(t, d) = tf(t, d) * log(N/(df + 1))$$")
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#Membuat Dataframe
dataTextPre = pd.read_csv('Prepocessing.csv',index_col=False)
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['Abstrak'])
# dataTextPre

"""### Matrik VSM(Visual Space Model)
Sebelum menghitung nilai TF, terlebih dahulu buat matrik vsm untuk menentukan bobot nilai term pada dokumen, hasilnya sebagaii berikut.
"""
matrik_vsm = bag.toarray()
matrik_vsm.shape
matrik_vsm[0]

a=vectorizer.get_feature_names()
# print(len(matrik_vsm[:,1]))

dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF.to_csv('TF.csv',index=False)
# st.table(dataTF)
st.dataframe(dataTF)

"""### Nilai Term Dokumen
Setelah didapat nilai matrik vsm, maka nilai term frequency yang didapat pada masing masing dokumen ialah seperti berikut.
"""
datalabel = pd.read_csv('Prepocessing.csv',index_col=False)
TF = pd.read_csv('TF.csv',index_col=False)
dataJurnal = pd.concat([TF, datalabel["Abstrak"]], axis=1)
# st.table(dataJurnal)
st.dataframe(dataJurnal)

st.subheader("Kmeans")
st.markdown("""
K-Means Clustering adalah suatu metode penganalisaan data atau metode Data Mining yang melakukan proses pemodelan unssupervised learning dan menggunakan metode yang mengelompokan data berbagai partisi.
K Means Clustering memiliki objective yaitu meminimalisasi object function yang telah di atur pada proses clasterisasi. Dengan cara minimalisasi variasi antar 1 cluster dengan maksimalisasi variasi dengan data di cluster lainnya.
K means clustering merupakan metode algoritma dasar,yang diterapkan sebagai berikut: 
Menentukan jumlah cluster
###a. Secara acak mendistribusikan data cluster
###b. Menghitung rata rata dari data yang ada di cluster.
###c. Menggunakan langkah baris 3 kembali sesuai nilai treshold
###d. Menghitung jarak antara data dan nilai centroid(K means clustering)
###e. Distance space dapat diimplementasikan untuk menghitung jarak data dan centroid. Contoh penghitungan jarak yang sering digunakan adalah manhattan/city blok distance
""")
st.latex("$$d(p,q) = \sqrt{(p_{1}-q_{1})^2+(p_{2}-q_{2})^2+(p_{3}-q_{3})^2}$$")
# d = jarak obyek
# p = data 
# q = centroid

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
# Latih Kmeans dengan n cluster terbaik
modelKm = KMeans(n_clusters=2, random_state=12)
modelKm.fit(dataTF.values)
prediksi = modelKm.predict(dataTF.values)

# Pengurangan dimensi digunakan untuk memplot dalam representasi 2d
pc=TruncatedSVD(n_components=2)
X_new=pc.fit_transform(dataTF.values)
centroids=pc.transform(modelKm.cluster_centers_)
centroids

dataTF['Cluster_Id'] = modelKm.labels_
dataTF

fig, ax = plt.subplots()
ax.scatter(X_new[:,0],X_new[:,1],c=prediksi, cmap='viridis')
ax.scatter(centroids[:,0] , centroids[:,1] , s = 50, color = 'red')
    
plt.tight_layout()
st.write(fig)