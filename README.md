# Kardiyak MRI Yeniden Oluşturma Görevi (CMR Reconstruction Challenge)

Bu repository, kardiyak MRI yeniden oluşturma görevleri için oluşturulmuş modelleri içermektedir. Bu challenge, farklı merkezlerden, farklı hastalıklardan ve farklı manyetik alan güçlerinden elde edilen kardiyak MRI görüntülerinin yeniden oluşturulması üzerine odaklanmaktadır.

## Görev 1: Çoklu Merkez Genellemesi

### Orijinal Görev Tanımı (İngilizce)

**TASK 1: CMR reconstruction model generalization for multiple centers**

This task primarily focuses on addressing the issue of declining generalization performance between multiple centers. Participants are required to train the reconstruction model on the training dataset and achieve good multi-contrast cardiac image reconstruction results on the validation and test datasets. It is important to note that for this task, we will include data from two entirely new centers in the validation set (not present in the training set), and the test set will contain data from five entirely new centers (not present in the training set, including the two centers that appeared in the validation set).

### Mala Anlatır Gibi (MAG)

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

### Mala Anlatır Gibi (MAG)

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

## Özel Görev 1: 5T Değerlendirmesi

### Orijinal Görev Tanımı (İngilizce)

**Special TASK 1: CMR reconstruction model for 5T evaluation**

This task primarily focuses on addressing the issue of declining reconstruction generalization performance under different magnetic field strengths, especially those not included in the training data. Participants are required to train the reconstruction model on the training dataset (mainly consisting of 1.5T and 3.0T) and achieve good multi-contrast cardiac image reconstruction results on the validation and test datasets (5.0T).

### Mala Anlatır Gibi (MAG)

#### 5T Değerlendirmesi için Kardiyak Görüntü Yeniden Oluşturma Görevi

Düşün ki bir fotoğraf makinesi var, ama bu makine farklı güçlerde çekim yapabiliyor. Bazı çekimler daha düşük güçte (1.5T ve 3.0T), bazıları ise çok daha yüksek güçte (5.0T).

Bilgisayarımızı sadece düşük güçteki çekimlerle eğitiyoruz. Ama sonra ondan yüksek güçteki çekimleri de anlamasını istiyoruz.

Bu biraz şuna benziyor: Diyelim ki sadece loş ışıkta çekilmiş fotoğrafları görmeye alıştın. Sonra birden çok parlak güneş ışığında çekilmiş bir fotoğrafı anlamaya çalışıyorsun. Görüntü aynı şeyi gösteriyor, ama çok farklı görünüyor!

İşte bu görevde:
- Bilgisayarı sadece 1.5T ve 3.0T güçlerindeki MRI görüntüleriyle eğitiyoruz
- Sonra bilgisayarın hiç görmediği 5.0T gücündeki görüntüleri ne kadar iyi işleyebildiğini test ediyoruz

Buradaki asıl zorluk: Bilgisayarın farklı manyetik alan güçlerindeki görüntüleri işleyebilmeyi öğrenmesi.

## Özel Görev 2: Pediatrik Görüntüleme Değerlendirmesi

### Orijinal Görev Tanımı (İngilizce)

**Special TASK 2: CMR reconstruction model for pediatric imaging evaluation**

This task primarily focuses on addressing application issues in pediatric cardiac imaging. Participants are required to train the reconstruction model on the training dataset (mainly consisting of adults over 20 years old) and achieve good multi-contrast cardiac image reconstruction results on the validation and test datasets (minors under 18 years old). Please note that to ensure the model training process is not biased by age information, we will not disclose the age of each data point in the training dataset.

### Mala Anlatır Gibi (MAG)

#### Pediatrik Görüntüleme için Kardiyak Görüntü Yeniden Oluşturma Görevi

Düşün ki yetişkin insanların yüzlerini tanımayı öğreniyorsun. Sonra birden çocuk yüzlerini de tanıman gerekiyor. Çocukların yüz özellikleri yetişkinlerden farklı - daha yuvarlak yüzler, daha büyük gözler ve farklı oranlar.

Kalp görüntüleri için de benzer bir durum var. Çocuk kalpleri, yetişkin kalplerinden farklı özelliklere sahiptir.

Bu görevde:
- Bilgisayarımızı sadece yetişkin kalp görüntüleriyle (20 yaş üstü) eğitiyoruz
- Bilgisayara kimin yaşlı kimin genç olduğunu söylemiyoruz
- Sonra bilgisayarın hiç görmediği çocuk kalp görüntülerini (18 yaş altı) ne kadar iyi işleyebildiğini test ediyoruz

Bu şuna benziyor: Sadece büyük elmalar görerek elmaları tanımayı öğrendin. Şimdi küçük elmaları da tanıyabilecek misin?

Buradaki asıl zorluk: Bilgisayarın yaşa bağlı anatomik farklılıklar olduğu halde kardiyak görüntüleri doğru şekilde işleyebilmesi.

## Değerlendirme Metrikleri ve Sıralama

### Orijinal Tanım (İngilizce)

**Metrics & Ranking**

**Metrics:**
For the four tasks, we will use SSIM, PSNR, and NMSE as objective evaluation metrics, with SSIM being the primary metric for ranking. During the test phase, we will conduct an additional round of scoring (1 to 5 points) by experienced radiologists for the top 5 teams in each task, and subsequent evaluations will be based on these subjective scores. The scoring will cover three aspects: image quality, image artifacts, and clinical utility. The final ranking will be determined by the average of the objective and subjective score rankings, and this average ranking will serve as the final competition ranking. Please note that during the validation phase, we will only conduct objective scoring and will not involve radiologist scoring.

**Ranking methods:**
1. When evaluating SSIM, we will narrow down the assessment field-of-view to the region where the heart is located, to avoid interference from the background area.
2. Regarding the sampling patterns and accelerations, all paired data are assigned equal weight when calculating the final ranking metrics.
3. Participating teams are required to submit docker containers and process all the cases in the test set on our server. For the cases without valid output, we will assign it to the lowest value of metric.

### Mala Anlatır Gibi (MAG)

#### Değerlendirme Nasıl Yapılacak?

**Kullanılacak Ölçümler:**
- Bilimsel ölçümler: SSIM (ana ölçüt), PSNR ve NMSE kullanılacak
- SSIM nedir? İki resmin ne kadar benzer olduğunu ölçen bir değer (1: tamamen aynı, 0: hiç benzerlik yok)
- Test aşamasında, her görevde en iyi 5 takımın sonuçları ayrıca deneyimli radyologlar tarafından puanlanacak (1-5 arası)
- Radyologlar üç şeye bakacak: görüntü kalitesi, görüntüdeki bozulmalar ve klinik kullanışlılık
- Son sıralama, bilimsel ölçümlerdeki sıra ile radyolog değerlendirmesindeki sıranın ortalaması olacak
- Doğrulama aşamasında sadece bilimsel ölçümler kullanılacak (radyolog değerlendirmesi olmayacak)

**Sıralama Yöntemleri:**
1. SSIM hesaplanırken sadece kalbin bulunduğu bölge değerlendirilecek (arka plan dahil edilmeyecek)
2. Farklı örnekleme desenleri ve hızlandırmalar içeren tüm eşleştirilmiş veriler, son sıralama metriklerinde eşit ağırlığa sahip olacak
3. Katılımcı takımların docker container'ları göndermesi ve test setindeki tüm vakaları bizim sunucumuzda işlemesi gerekiyor
4. Geçerli çıktı üretilmeyen vakalar için en düşük ölçüm değeri atanacak

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
