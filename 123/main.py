import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загружаем изображение
img = cv2.imread('text.jpg', cv2.IMREAD_GRAYSCALE)

# Применяем пороговую обработку
_, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

# Сохраняем результат
cv2.imwrite('text.jpg', thresh)
# Загружаем пороговое изображение
thresh = cv2.imread('text.jpg', cv2.IMREAD_GRAYSCALE)

# Создаем случайный шум
noise = np.zeros(thresh.shape, np.uint8)
cv2.randn(noise, 0, 50)

# Добавляем шум на изображение
noisy = cv2.add(thresh, noise)

# Сохраняем результат
cv2.imwrite('noisy_text.png', noisy)

# Применяем пороговую обработку на зашумленном изображении
_, thresh = cv2.threshold(noisy, 150, 255, cv2.THRESH_BINARY)

# Сохраняем результат
cv2.imwrite('final_text.png', thresh)


plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(thresh),plt.title('Output')
plt.show()