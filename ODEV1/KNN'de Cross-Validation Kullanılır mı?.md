# KNN ve Cross-Validation

## KNN'de Cross-Validation Kullanılır mı?

Evet. K-Nearest Neighbors (KNN) algoritmasında cross-validation (çapraz doğrulama) hem mümkün hem de sıkça kullanılır. KNN'de hiperparametre seçimi (özellikle **k** değeri ve mesafe metriği) model performansını doğrudan etkilediği için, bu seçimi cross-validation ile yapmak iyi bir pratiktir.

## Neden KNN'de Cross-Validation?

- **k seçimi:** k çok küçükse overfitting, çok büyükse underfitting riski vardır. Cross-validation ile farklı k değerleri denenir ve en iyi k seçilir.
- **Veri verimliliği:** KNN genelde küçük/orta veri setlerinde kullanılır. Tüm veriyi eğitim için kullanıp ayrı bir test seti bırakmak yerine, cross-validation ile verinin farklı bölümleri sırayla test olarak kullanılır; böylece daha güvenilir performans tahmini elde edilir.
- **Varyans azaltma:** Tek bir train-test bölmesi şansa bağlı olabilir. K-fold cross-validation ile birden fazla bölme kullanıldığı için tahminler daha kararlı olur.

## Nasıl Uygulanır?

**K-Fold Cross-Validation** tipik akış:

- Veri seti K eşit parçaya (fold) bölünür (örn. K=5 veya K=10).
- Her fold sırayla test, diğer K−1 fold eğitim olarak kullanılır.
- Her fold için bir doğruluk (veya başka metrik) hesaplanır.
- K fold'un ortalaması alınarak genel performans ve k seçimi değerlendirilir.
- Aynı süreç, farklı k değerleri (örn. 1, 3, 5, 7, …, 21) için tekrarlanır; en yüksek ortalama doğruluğu veren k seçilir.

## Özet

- KNN'de cross-validation yapılır ve yapılmalıdır.
- En yaygın kullanım amacı, en uygun **k** (ve isteğe bağlı olarak mesafe metriği) değerini veriye göre seçmek ve model performansını daha güvenilir tahmin etmektir.
- Scikit-learn'de **GridSearchCV** veya **cross_val_score** ile KNN + cross-validation kolayca uygulanabilir.
