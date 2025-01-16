

# Görüntü Netleştirme Projesi (Image Deblurring)

Bu proje, bulanıklaştırılmış görüntüleri netleştirmek için bir derin öğrenme modeli (U-Net) kullanır. Proje, MATLAB ortamında geliştirilmiştir ve bulanık görüntüleri gerçek keskin görüntülere dönüştürmeyi amaçlar.

## Proje İçeriği

Proje, aşağıdaki bileşenleri içerir:

1. **Veri Seti Hazırlama**:
   
   - Bulanık ve keskin görüntü çiftlerinden oluşan bir veri seti kullanılır.
   - Veri seti, eğitim ve test olarak ikiye ayrılır.
   - Görüntüler, modelin giriş boyutuna uygun şekilde yeniden boyutlandırılır ve normalize edilir.

2. **Veri Artırma (Data Augmentation)**:
   
   - Eğitim verilerine rastgele döndürme, yansıma ve ölçeklendirme gibi işlemler uygulanarak modelin genelleme yeteneği artırılır.

3. **U-Net Modeli**:
   
   - U-Net mimarisi kullanılarak bir derin öğrenme modeli oluşturulur.
   - Model, bulanık görüntüleri keskinleştirmek için eğitilir.

4. **Eğitim ve Test**:
   
   - Model, eğitim verileri üzerinde eğitilir ve test verileri üzerinde değerlendirilir.
   - Eğitim süreci sırasında kayıp (loss) ve performans metrikleri görselleştirilir.

5. **Sonuçların Görselleştirilmesi**:
   
   - Test aşamasında, rastgele seçilen bulanık görüntüler üzerinde modelin tahminleri görselleştirilir.
   - Bulanık, gerçek keskin ve model tarafından keskinleştirilmiş görüntüler yan yana gösterilir.

## Nasıl Çalıştırılır?

1. **Gereksinimler**:
   
   - MATLAB (R2020a veya üzeri önerilir).
   - Deep Learning Toolbox.
   - İmage Processing Toolbox

2. **Veri Seti**:
   
   - Projeyi çalıştırmak için bulanık ve keskin görüntü çiftlerinden oluşan bir veri setine ihtiyacınız var.
   - Projede HIDE veri setini kullandım.
   - Veri seti, `test/blur` ve `test/sharp` klasörlerinde bulunmalıdır.

3. **Kodu Çalıştırma**:
   
   - MATLAB'de `main.m` dosyasını açın ve çalıştırın.
   - Kod, veri setini yükleyecek, modeli eğitecek ve test edecektir.

4. **Modeli Kaydetme**:
   
   - Eğitilen model, `trainedUNetModel.mat` dosyası olarak kaydedilir.

5. **Sonuçları Görselleştirme**:
   
   - Test aşamasında, rastgele seçilen görüntüler üzerinde modelin performansı görselleştirilir.

## Kullanılan Teknolojiler

- **MATLAB**: Projenin geliştirildiği ana platform.
- **Deep Learning Toolbox**: Derin öğrenme modelinin oluşturulması ve eğitimi için kullanıldı.
- **U-Net**: Görüntü keskinleştirme için kullanılan derin öğrenme mimarisi.

## Proje Yapısı

image-deblurring-project/  
│  
├── test/  
│ ├── blur/ # Bulanık görüntüler  
│ └── sharp/ # Keskin görüntüler  
│  
├── main.m # Ana proje dosyası  
├── trainedUNetModel.mat # Eğitilmiş model dosyası  
└── README.md # Proje dokümantasyonu
