import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


LABEL_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


current_dir = os.path.dirname(os.path.abspath(__file__))

print("CIFAR‑10 k‑NN sınıflandırma örneği")
print("-----------------------------------")

print(
    "Lütfen CIFAR‑10 Python versiyonunun (cifar-10-batches-py klasörü) "
    "bu dizinin altına yerleştirildiğinden emin olun:\n"
    f"{current_dir}"
)

cifar_dir = os.path.join(current_dir, "cifar-10-batches-py")

if not os.path.isdir(cifar_dir):
    raise FileNotFoundError(
        f"'cifar-10-batches-py' klasörü bulunamadı.\n"
        f"Lütfen CIFAR‑10 Python arşivini indirip açın ve klasörü buraya koyun:\n{cifar_dir}"
    )


def load_batch(filepath):
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]       # shape: (N, 3072)
    labels = batch[b"labels"]   # uzunluk N
    data = data.astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    return data, labels


print("\nVeri yükleniyor...")

train_data_list = []
train_labels_list = []

for i in range(1, 6):
    batch_path = os.path.join(cifar_dir, f"data_batch_{i}")
    data, labels = load_batch(batch_path)
    train_data_list.append(data)
    train_labels_list.append(labels)

train_data = np.concatenate(train_data_list, axis=0)   
train_labels = np.concatenate(train_labels_list, axis=0)  

test_batch_path = os.path.join(cifar_dir, "test_batch")
test_data, test_labels = load_batch(test_batch_path)   

print(f"Eğitim veri boyutu: {train_data.shape}")
print(f"Test veri boyutu  : {test_data.shape}")


max_train = 5000   
max_test = 50      

if train_data.shape[0] > max_train:
    train_data = train_data[:max_train]
    train_labels = train_labels[:max_train]

if test_data.shape[0] > max_test:
    test_data = test_data[:max_test]
    test_labels = test_labels[:max_test]

print(f"Kullanılan eğitim örnek sayısı: {train_data.shape[0]}")
print(f"Kullanılan test   örnek sayısı: {test_data.shape[0]}")


print("\nUzaklık tipi seçin:")
print("1: L1 (Manhattan)")
print("2: L2 (Öklid)")

distance_choice = input("Seçiminiz (1 veya 2): ").strip()

if distance_choice == "1":
    use_l1 = True
    print("Seçilen uzaklık: L1 (Manhattan)")
elif distance_choice == "2":
    use_l1 = False
    print("Seçilen uzaklık: L2 (Öklid)")
else:
    raise ValueError("Geçersiz seçim. Lütfen sadece 1 veya 2 girin.")


k_str = input("k değerini girin (pozitif tam sayı): ").strip()

if not k_str.isdigit():
    raise ValueError("k değeri pozitif bir tam sayı olmalıdır.")

k = int(k_str)

if k <= 0:
    raise ValueError("k değeri 0'dan büyük olmalıdır.")

print(f"Seçilen k: {k}")


index_str = input(
    f"Sınıflandırmak istediğiniz test örneğinin indeksini girin (0-{test_data.shape[0]-1}): "
).strip()

if not index_str.isdigit():
    raise ValueError("İndeks pozitif bir tam sayı olmalıdır.")

test_index = int(index_str)

if not (0 <= test_index < test_data.shape[0]):
    raise ValueError("Geçersiz indeks aralığı.")

test_sample = test_data[test_index]      
true_label = test_labels[test_index]     

print(f"Seçilen test örneği indeks: {test_index}")
print(f"Gerçek etiket: {true_label} ({LABEL_NAMES[true_label]})")


print("\nUzaklıklar hesaplanıyor, lütfen bekleyin...")

if use_l1:
    distances = np.sum(np.abs(train_data - test_sample), axis=1)
else:
    diff = train_data - test_sample
    distances = np.sum(diff * diff, axis=1)


nearest_indices = np.argsort(distances)[:k]
nearest_labels = train_labels[nearest_indices]

values, counts = np.unique(nearest_labels, return_counts=True)
pred_label = values[np.argmax(counts)]


print("\nSonuçlar")
print("--------")
print(f"Tahmin edilen etiket: {pred_label} ({LABEL_NAMES[pred_label]})")
print(f"Gerçek etiket      : {true_label} ({LABEL_NAMES[true_label]})")


show_str = input("Görüntüyü göstermek ister misiniz? (e/h): ").strip().lower()

if show_str == "e":
    img = test_sample.reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0)) / 255.0

    plt.imshow(img)
    plt.title(
        f"Tahmin: {LABEL_NAMES[pred_label]} | Gerçek: {LABEL_NAMES[true_label]}"
    )
    plt.axis("off")
    plt.show()

print("\nProgram sona erdi.")

