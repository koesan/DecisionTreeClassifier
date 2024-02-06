import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Veri setindeki verileri data değişkenine aktar
data = pd.read_csv('breast-cancer.csv')

# Veri setini bölmek için Özellikleri(belirtileri) x etiketleri(teşhisleri) ise y değişkenine ekledi id 
x = data.drop(['diagnosis', 'id'] , axis=1)
y = data[['diagnosis']]

# Veriler aynı formatta olması gerekir x te hepsi float olması lazım int gibi değişkenler(id) varsa çıkarılır

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Karar ağacı sınıflandırıcısını oluştur
dtree = DecisionTreeClassifier()

# Veri seti ile karar ağacını eğit
dtree.fit(x_train, y_train)

# X_test verisine göre sınıflandırma yap
Y_pred = dtree.predict(x_test)

# Sonuçları bastır
for i in range(len(Y_pred)):
	print(f"Etiket = {y_test.iloc[i]}, Tahmin = {Y_pred[i]}\n")

# Tek bir veri üstünden sınıflandırma yapma
index = 19
X_test = x_test.iloc[[index]]

# X_test verisine göre sınıflandırma yap
Y_pred = dtree.predict(X_test)

# Sonucu bastır
print(f"{index}.Etiket = {y_test.iloc[index]}, {index}.Tahmin = {Y_pred[0]}\n")

# Modelin başarı skoru
print(f"Başarı: {dtree.score(x_test, y_test)} \n")