# Kardiyak MRI Yeniden Oluşturma Görevi (CMR Reconstruction Challenge)

Bu repository, kardiyak MRI yeniden oluşturma görevleri için oluşturulmuş modelleri içermektedir. Bu challenge, farklı merkezlerden ve farklı hastalıklardan elde edilen kardiyak MRI görüntülerinin yeniden oluşturulması üzerine odaklanmaktadır.

## Görev 1: Çoklu Merkez Genellemesi (bunu seçebiliriz)

### Orijinal Görev Tanımı (İngilizce)

**TASK 1: CMR reconstruction model generalization for multiple centers**

This task primarily focuses on addressing the issue of declining generalization performance between multiple centers. Participants are required to train the reconstruction model on the training dataset and achieve good multi-contrast cardiac image reconstruction results on the validation and test datasets. It is important to note that for this task, we will include data from two entirely new centers in the validation set (not present in the training set), and the test set will contain data from five entirely new centers (not present in the training set, including the two centers that appeared in the validation set).

### Mala Anlatır Gibi

#### Kardiyak Görüntü Yeniden Oluşturma Görevi

Düşün ki bir oyun oynuyoruz. Bu oyunda, kalp resimlerini toplayıp bir bilgisayarı eğitiyoruz. 

Bilgisayarımızı bazı hastanelerden topladığımız resimlerle eğitiyoruz. Ama sonra, bilgisayarın hiç görmediği hastanelerden gelen resimleri de tanımasını istiyoruz. Bu biraz zor, çünkü her hastane resimleri farklı şekilde çekebilir.

İşte görev:
- Önce bilgisayarı bazı hastanelerden gelen resimlerle eğit
- Sonra bu bilgisayarın, hiç görmediği iki yeni hastaneden gelen resimlerde nasıl çalıştığını göreceğiz (bu deneme aşaması)
- En son, bilgisayarın beş yeni hastaneden gelen resimlerde nasıl çalıştığına bakacağız (final sınav gibi)

Bu biraz, bir arkadaşının ağzını kapatıp konuşmasını anlamaya çalışmak gibi. Önce bazı kişilerin kapalı ağızla konuşmalarını dinleyip öğreniyorsun, sonra hiç duymadığın kişilerin kapalı ağızla konuşmalarını anlamaya çalışıyorsun.

Buradaki asıl zorluk: Bilgisayarın sadece belli hastanelerin çekim stilini öğrenmesi değil, genel olarak kalp resimlerini anlamayı öğrenmesi.

## Görev 2: Çoklu Hastalık Değerlendirmesi

### Orijinal Görev Tanımı (İngilizce)

**Regular TASK 2: CMR reconstruction model for multiple diseases evaluation**

This task primarily focuses on evaluating the reliability of the model in applications involving different cardiovascular diseases. Participants are required to train the reconstruction model on the training dataset and achieve good performance in disease applications on the validation and test datasets. It is important to note that for this task, we will include data for two diseases that have not appeared in the training set in the validation set, and the test set will contain data for five diseases that have not appeared in the training set (including the two diseases that appeared in the validation set). Please note that to ensure the model training process is not biased by the type of disease, we will not disclose the disease information for each data point in the training and validation dataset.

### Mala Anlatır Gibi

#### Kalp Hastalıkları Tanıma Görevi

Düşün ki bir doktor olmak için çalışıyorsun. Önce bazı hastalıkları tanımayı öğreniyorsun. Sonra hiç görmediğin hastalıkları da tanıman gerekiyor.

Bu görevde:
- Bilgisayarımızı bazı kalp hastalıklarının görüntüleriyle eğitiyoruz
- Ama bilgisayara "bu hastalık budur" diye söylemiyoruz, sadece görüntüleri gösteriyoruz
- Sonra bilgisayarın hiç görmediği iki yeni hastalığın görüntülerini nasıl işlediğini test ediyoruz (deneme aşaması)
- En son, beş yeni hastalığın görüntüleriyle final sınavı yapıyoruz

Bu şuna benziyor: Diyelim ki sen sadece elma ve armut görerek meyveleri tanımayı öğrendin. Sonra birden karşına hiç görmediğin bir muz çıkıyor. "Bu da bir meyve mi, nasıl anlayacağım?" diye düşünüyorsun.

İşte buradaki zorluk da bu: Bilgisayar sadece belirli hastalıkların özelliklerini değil, genel olarak kalp görüntülerinde hastalık işaretlerini anlayabilmeli.

Ve en ilginç kısmı, bilgisayara hangi hastalığın ne olduğunu söylemiyoruz! Çünkü gerçek hayatta da bazen karşımıza çok nadir görülen hastalıklar çıkabilir.

## Kurulum ve Kullanım

(Bu bölümde projenin nasıl kurulacağı ve kullanılacağı hakkında bilgiler eklenecektir.)

## Veri Seti

(Bu bölümde veri seti hakkında bilgiler eklenecektir.)

## Model Mimarisi

(Bu bölümde kullanılan model mimarisi hakkında bilgiler eklenecektir.)

## Sonuçlar

(Bu bölümde elde edilen sonuçlar paylaşılacaktır.)

## Lisans

(Bu bölümde lisans bilgisi eklenecektir.)

## İletişim

(Bu bölümde iletişim bilgileri eklenecektir.)
