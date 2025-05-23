# Evcil Hayvan Tanıma Sistemi (Kedi/Köpek)

Evcil Hayvan Tanıma Sistemi (Kedi/Köpek), görüntüler veya video akışı üzerinden kedi ve köpekleri otomatik olarak tespit eden bir yapay zekâ uygulamasıdır. Bu sistem, YOLOv5 gibi derin öğrenme tabanlı nesne 
tanıma algoritmalarını kullanarak, evcil hayvanların konumlarını belirler ve görsel çıktılarla etiketlenmiş olarak sunar. Barınaklarda, veteriner kliniklerinde veya akıllı güvenlik sistemlerinde kullanıma uygundur.


# Projenin Amacı

Bu proje, görsel ya da video görüntüler üzerinden kedileri ve köpekleri doğru bir şekilde tespit eden yapay zekâ tabanlı bir sistem geliştirmeyi amaçlar. 
Sistem, barınaklar, veteriner klinikleri ve güvenlik sistemlerinde kullanılmak üzere tasarlanabilir.


#  Hedef Kullanıcı Kitlesi

-Veteriner klinikleri
-Hayvan barınakları
-Hayvan sahipleri
-Yapay zekâ ve görüntü işleme öğrenmek isteyen öğrenciler


# Kullanılan Teknolojiler / Algoritmalar

-Python
-OpenCV
-YOLOv5 / YOLOv8 (önceden eğitilmiş model)
-PyTorch
-NumPy


# Kurulum Adımları

git clone (https://github.com/AlptekinZengin/Yapay-Zeka-ile-Ak-ll-Tespit-ve-Tan-ma-Projesi/blob/main/Evcil%20Hayvan%20Tan%C4%B1ma%20Sistemi%20(kedi%2C%20k%C3%B6pek))

cd evcil-hayvan-tanima
pip install -r requirements.txt
python detect.py --source "veri/test.jpg"
