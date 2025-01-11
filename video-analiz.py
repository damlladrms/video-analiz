import cv2
import pytesseract
from datetime import datetime
import face_recognition
import numpy as np  # Standart sapma için numpy kütüphanesi
import os  # Dosya işlemleri için os kütüphanesi

# Video dosyasını yükleyin
video_path = 'video_dosyanız.mp4'
cap = cv2.VideoCapture(video_path)

# OCR ayarları
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Tesseract yolunu belirtin

# Yüz tanıma için çalışan yüz veritabanı
known_face_encodings = []
known_face_names = ["Çalışan 1", "Çalışan 2", "Çalışan 3"]  # Çalışanların isimleri

# Çalışanların yüz resimlerini yükleyin ve kodlayın
image_of_employee1 = face_recognition.load_image_file("calisan1_resmi.jpg")
employee1_face_encoding = face_recognition.face_encodings(image_of_employee1)[0]
known_face_encodings.append(employee1_face_encoding)

image_of_employee2 = face_recognition.load_image_file("calisan2_resmi.jpg")
employee2_face_encoding = face_recognition.face_encodings(image_of_employee2)[0]
known_face_encodings.append(employee2_face_encoding)

image_of_employee3 = face_recognition.load_image_file("calisan3_resmi.jpg")
employee3_face_encoding = face_recognition.face_encodings(image_of_employee3)[0]
known_face_encodings.append(employee3_face_encoding)

# Çalışanların karelerini kaydetmek için bir klasör oluşturun
output_folder = "kaydedilen_goruntuler"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Zamanı almak için kullanılan fonksiyon
def get_current_time_from_frame(frame):
    # OCR ile saati tanıyın
    text = pytesseract.image_to_string(frame, config='--psm 6')
    return text

# Çalışanların çalışma sürelerini saklamak için bir sözlük
work_durations = {name: [] for name in known_face_names}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Yüzleri tanıma işlemi
    rgb_frame = frame[:, :, ::-1]  # BGR'den RGB'ye dönüştür
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:  # Eğer yüz tanındıysa
            name = known_face_names[matches.index(True)]
            # Yüzün etrafında bir dikdörtgen çizin
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Görüntüyü kaydetme
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = os.path.join(output_folder, f"{name}_{timestamp}.png")
            cv2.imwrite(image_filename, frame)

            # Zamanı kaydetme
            current_time = get_current_time_from_frame(frame)
            if "Başlangıç" in current_time:  # Başlangıç işareti
                start_time = datetime.now()
            elif "Bitiş" in current_time:  # Bitiş işareti
                end_time = datetime.now()
                work_duration = (end_time - start_time).total_seconds()  # Süreyi saniye cinsinden hesapla
                work_durations[name].append(work_duration)  # Çalışanın süresini kaydet
                break

    # Frame'leri göster
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Her bir çalışanın standart sapmalarını hesaplama ve %10 performans kaybını ekleme
results = []
for name, durations in work_durations.items():
    if len(durations) > 0:
        mean_duration = np.mean(durations)  # Ortalama çalışma süresi
        std_deviation = np.std(durations)  # Çalışma süresi standart sapması
        # Performans kaybını ekleme
        adjusted_std_deviation = std_deviation * 1.10  # %10 performans kaybı ekle
        results.append([name, mean_duration, adjusted_std_deviation])
        print(f"{name} için ortalama çalışma süresi: {mean_duration:.2f} saniye")
        print(f"{name} için düzeltilmiş standart sapma: {adjusted_std_deviation:.2f} saniye")
    else:
        results.append([name, None, None])
        print(f"{name} için çalışma süresi tespiti başarısız oldu.")

# Kullanıcıya sonuçları kaydetmek isteyip istemediğini soralım
def save_results_to_csv(results):
    filename = 'calisanlar_sonuclari.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Çalışan Adı", "Ortalama Çalışma Süresi (saniye)", "Düzeltilmiş Standart Sapma"])
        for result in results:
            writer.writerow(result)
    print(f"Sonuçlar {filename} dosyasına kaydedildi.")

# Kullanıcıya kaydetmek isteyip istemediğini soralım
save_option = input("Sonuçları kaydetmek ister misiniz? (Evet/Hayır): ").strip().lower()
if save_option == 'evet':
    save_results_to_csv(results)
else:
    print("Sonuçlar kaydedilmedi.")
