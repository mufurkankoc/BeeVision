# BeeVision
 BeeVision teknik mülakatı için hazırlanmış repo
 
 
# Kullanım
 - python3 beevision.py --image filename
 - Ör: python3 beevision.py --image image.jpg
 
 
 # İçerik
  - İstenilen imaj dosyası argparse ile içe aktarılabilmektedir.
  - İmaj üzerinde bulunan etiketler tespit edilir, bbox çizilir, conf. değeri ile birlikte gösterilir.
  - Birden fazla etiket var ise onları da bulmaktadır.
  - Tespit edilen etiketin köşegen lokasyonları sonrasında kullanılmak üzere return edilir.
  - Buna örnek olarak 1 etiket var ise ROI bölgesi kesilerek ayrı bir dosya olarak kaydedilir.
  - 1'den fazla etiket var ise ROI bölgelerinin köşegen lokasyonları bir liste şeklinde print edilir.


 # Kütüphaneler
  - torch
  - numpy as np
  - cv2
  - ultralytics 
  - supervision
  - argparse
