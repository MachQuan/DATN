Hướng dẫn sử dụng chương trình nhận diện biển báo giao thông bằng Python và OpenCV
# 1. Cài đặt các thư viện cần thiết
Trước tiên, bạn cần cài đặt các thư viện cần thiết. Bạn có thể sử dụng pip để cài đặt OpenCV và các thư viện liên quan khác.

bash
Sao chép mã
pip install opencv-python numpy
# 2. Chuẩn bị dữ liệu
Bạn cần có tập dữ liệu về các biển báo giao thông để huấn luyện và kiểm tra mô hình. Tập dữ liệu có thể bao gồm các hình ảnh biển báo và nhãn tương ứng. Có nhiều tập dữ liệu sẵn có trên mạng như GTSRB (German Traffic Sign Recognition Benchmark).

# 3. Tạo chương trình nhận diện biển báo
Dưới đây là mã nguồn cơ bản để đọc hình ảnh, tiền xử lý và nhận diện biển báo giao thông.

python
Sao chép mã
import cv2
import numpy as np

# Hàm để tải và tiền xử lý ảnh
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    normalized = resized / 255.0
    return normalized

# Hàm để nhận diện biển báo giao thông
def recognize_traffic_sign(image_path, model):
    processed_image = preprocess_image(image_path)
    processed_image = processed_image.reshape(1, 32, 32, 1)
    
    prediction = model.predict(processed_image)
    class_id = np.argmax(prediction)
    
    return class_id

# Tải mô hình đã huấn luyện (ví dụ: sử dụng mô hình Keras)
from keras.models import load_model
model = load_model('traffic_sign_model.h5')

# Nhận diện biển báo giao thông từ hình ảnh
image_path = 'path_to_traffic_sign_image.jpg'
class_id = recognize_traffic_sign(image_path, model)

# In ra kết quả
print(f'Biển báo giao thông được nhận diện là: {class_id}')
# 4. Huấn luyện mô hình nhận diện (nếu chưa có mô hình đã huấn luyện)
Để huấn luyện mô hình, bạn cần sử dụng tập dữ liệu đã chuẩn bị và một framework học sâu như TensorFlow/Keras. Dưới đây là một ví dụ đơn giản về cách huấn luyện mô hình CNN để nhận diện biển báo.

python
Sao chép mã
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Chuẩn bị dữ liệu huấn luyện
def load_data():
    # Việc tải và tiền xử lý dữ liệu sẽ tùy thuộc vào tập dữ liệu cụ thể
    # Đây chỉ là ví dụ giả định
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    y_train = to_categorical(y_train, num_classes=43)
    y_test = to_categorical(y_test, num_classes=43)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(43, activation='softmax')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Lưu mô hình
model.save('traffic_sign_model.h5')
# 5. Chạy chương trình
Sau khi đã có mô hình huấn luyện, bạn có thể sử dụng nó để nhận diện các biển báo giao thông mới bằng cách sử dụng đoạn mã đã trình bày ở mục 3.

