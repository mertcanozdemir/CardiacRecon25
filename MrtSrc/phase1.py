import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class CMRxReconDataset(Dataset):
    """
    CMRxRecon veri seti için özel PyTorch Dataset sınıfı.
    Verileri belleğe önceden yüklemek yerine, gerektiğinde yükler.
    """
    def __init__(self, root_dir, file_paths_with_metadata, transform=None):
        self.root_dir = root_dir
        self.file_paths_with_metadata = file_paths_with_metadata
        self.transform = transform

    def __len__(self):
        return len(self.file_paths_with_metadata)

    def __getitem__(self, idx):
        relative_file_path, metadata = self.file_paths_with_metadata[idx]
        full_file_path = os.path.join(self.root_dir, relative_file_path)

        try:
            with h5py.File(full_file_path, 'r') as hf_m:
                kspace_real = hf_m['kspace']['real'][:]
                kspace_imag = hf_m['kspace']['imag'][:]
                kspace = kspace_real + 1j * kspace_imag
                
                # NumPy dizisini PyTorch tensörüne dönüştür
                kspace_tensor = torch.from_numpy(kspace).to(torch.complex64)

                # Maske verisini yükle (eğer varsa ve gerekiyorsa)
                # ShowCase.py'da maske ayrı bir dosyadan yükleniyordu.
                # Eğer her kspace dosyasıyla ilişkili bir maske varsa, buraya eklenmeli.
                # Şimdilik, sadece kspace'i döndürüyoruz.
                mask = None # Varsayılan olarak maske yok
                # Örnek: Eğer maske dosyası kspace dosyasıyla aynı dizinde ve benzer isimdeyse:
                # mask_file_path = full_file_path.replace('cine_sax.mat', 'cine_sax_mask_ktRadial8.mat') # Örnek maske ismi
                # if os.path.exists(mask_file_path):
                #     with h5py.File(mask_file_path, 'r') as mask_s:
                #         mask = torch.from_numpy(mask_s['mask'][:]).to(torch.float32)


            if self.transform:
                kspace_tensor = self.transform(kspace_tensor)

            return kspace_tensor, metadata, mask # Maskeyi de döndürebiliriz

        except Exception as e:
            print(f"Hata oluştu {full_file_path} yüklenirken: {e}")
            # Hata durumunda boş veya özel bir değer döndürebilirsiniz
            return None, None, None

def get_file_list_from_directory(root_dir, extension='.mat'):
    """
    Belirtilen kök dizindeki tüm .mat dosyalarının göreceli yollarını döndürür.
    """
    file_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(extension):
                full_path = os.path.join(dirpath, f)
                relative_path = os.path.relpath(full_path, root_dir)
                file_list.append(relative_path)
    return file_list

def prepare_datasets(root_dir, file_list, test_size=0.2, val_size=0.1, random_state=42):
    """
    Veri setini eğitim, doğrulama ve test olarak böler ve Dataset nesneleri döndürür.

    Args:
        root_dir (str): .mat dosyalarının bulunduğu kök dizin.
        file_list (list): İşlenecek .mat dosyalarının listesi (kök dizine göre göreceli yollar).
        test_size (float): Test seti için ayrılacak veri oranı.
        val_size (float): Doğrulama seti için ayrılacak veri oranı.
        random_state (int): Veri bölme için rastgele durum tohumu.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) olarak bölünmüş veri setleri.
    """
    all_file_paths_with_metadata = []

    print(f"{len(file_list)} dosya için meta veri çıkarılıyor...")

    for i, relative_file_path in enumerate(file_list):
        # Dosya yolundan meta veri çıkarma (örnek: Center, Siemens)
        # Örnek yol: ChallengeData/MultiCoil/Cine/TrainingSet/FullSample/Center005/Siemens_30T_Vida/P003/cine_sax.mat
        path_parts = relative_file_path.split(os.sep)
        metadata = {}
        metadata['file_path'] = relative_file_path
        
        # Yol yapısına göre meta veri çıkarma mantığını buraya ekleyin
        # ShowCase.py'daki örnek yola göre:
        # rootdir = "F:/CMRxRecon2025/ChallengeData/MultiCoil/Cine/TrainingSet/"
        # file_name = os.path.join(rootdir,"FullSample", "Center005","Siemens_30T_Vida","P003","cine_sax.mat")
        
        # Bu örnekte, path_parts'ın son 4 elemanı: P003, cine_sax.mat, Siemens_30T_Vida, Center005
        if len(path_parts) >= 4: 
            metadata['patient_id'] = path_parts[-2] # P003
            metadata['scanner_info'] = path_parts[-3] # Siemens_30T_Vida
            metadata['center_id'] = path_parts[-4] # Center005
        else:
            metadata['patient_id'] = 'unknown'
            metadata['scanner_info'] = 'unknown'
            metadata['center_id'] = 'unknown'

        all_file_paths_with_metadata.append((relative_file_path, metadata))

    # Test setini ayır
    train_val_files, test_files = train_test_split(
        all_file_paths_with_metadata, test_size=test_size, random_state=random_state
    )

    # Eğitim ve doğrulama setlerini ayır
    train_files, val_files = train_test_split(
        train_val_files, test_size=val_size/(1-test_size), random_state=random_state
    )

    print(f"\nVeri Bölme Sonuçları:")
    print(f"Eğitim Seti: {len(train_files)} dosya")
    print(f"Doğrulama Seti: {len(val_files)} dosya")
    print(f"Test Seti: {len(test_files)} dosya")

    train_dataset = CMRxReconDataset(root_dir, train_files)
    val_dataset = CMRxReconDataset(root_dir, val_files)
    test_dataset = CMRxReconDataset(root_dir, test_files)

    return train_dataset, val_dataset, test_dataset

# --- Kullanım Örneği ---
# Gerçek kullanımda, 'root_dir' kendi veri setinize göre ayarlanmalıdır.

if __name__ == '__main__':
    # Kendi veri setinizin kök dizinini buraya girin
    # Örneğin: root_directory = "/path/to/your/CMRxRecon_data/ChallengeData/MultiCoil/Cine/TrainingSet/"
    # VEYA ShowCase.py'daki gibi:
    root_directory = "mnt/f/CMRxRecon2025/ChallengeData/MultiCoil/Cine/TrainingSet/" # Bu yolu kendi sisteminize göre güncelleyin!

    # Tüm .mat dosyalarını otomatik olarak bul
    # Bu fonksiyonu kullanmadan önce root_directory'nin doğru olduğundan emin olun
    if os.path.exists(root_directory):
        all_mat_files = get_file_list_from_directory(root_directory)
        if not all_mat_files:
            print(f"Uyarı: '{root_directory}' dizininde hiç .mat dosyası bulunamadı. Lütfen yolu kontrol edin.")
            print("Örnek bir dosya listesi ile devam ediliyor...")
            # Test amaçlı örnek dosya listesi (gerçekte bu kısmı silmelisiniz)
            all_mat_files = [
                "FullSample/Center005/Siemens_30T_Vida/P003/cine_sax.mat",
                "FullSample/Center001/GE_15T_Signa/P001/cine_sax.mat",
                "FullSample/Center002/Philips_15T_Achieva/P002/cine_sax.mat",
                "FullSample/Center005/Siemens_30T_Vida/P004/cine_sax.mat",
                "FullSample/Center001/GE_15T_Signa/P005/cine_sax.mat",
            ]
    else:
        print(f"Hata: '{root_directory}' dizini bulunamadı. Lütfen doğru yolu girin.")
        print("Örnek bir dosya listesi ile devam ediliyor...")
        # Test amaçlı örnek dosya listesi (gerçekte bu kısmı silmelisiniz)
        all_mat_files = [
            "FullSample/Center005/Siemens_30T_Vida/P003/cine_sax.mat",
            "FullSample/Center001/GE_15T_Signa/P001/cine_sax.mat",
            "FullSample/Center002/Philips_15T_Achieva/P002/cine_sax.mat",
            "FullSample/Center005/Siemens_30T_Vida/P004/cine_sax.mat",
            "FullSample/Center001/GE_15T_Signa/P005/cine_sax.mat",
        ]

    train_dataset, val_dataset, test_dataset = prepare_datasets(root_directory, all_mat_files)

    # DataLoader oluşturma
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers'ı artırabilirsiniz
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nDataLoader'lar oluşturuldu. Batch boyutu: {batch_size}")
    print(f"Eğitim DataLoader'ı boyutu: {len(train_loader)}")
    print(f"Doğrulama DataLoader'ı boyutu: {len(val_loader)}")
    print(f"Test DataLoader'ı boyutu: {len(test_loader)}")

    # DataLoader'dan bir örnek alma
    print("\nDataLoader'dan bir örnek yükleniyor...")
    for i, (kspace_batch, metadata_batch, mask_batch) in enumerate(train_loader):
        if kspace_batch is not None:
            print(f"Batch {i+1}: K-space batch şekli: {kspace_batch.shape}")
            print(f"Batch {i+1}: İlk meta veri: {metadata_batch[0]}")
            # print(f"Batch {i+1}: Maske batch şekli: {mask_batch.shape if mask_batch is not None else 'Yok'}")
            break
        else:
            print(f"Batch {i+1}: Veri yüklenirken hata oluştu, bu batch atlanıyor.")

    print("\n--- Aşama 1 Veri Ön İşleme Betiği DataLoader ile Güncellendi ---")
    print("Lütfen `root_directory` değişkenini kendi veri setinizin kök dizinine göre güncelleyin.")
    print("Maske yükleme kısmı, veri setinizin yapısına göre özelleştirilmelidir.")


