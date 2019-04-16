# DSP_test
DSP test

Тестовое задание в DSP

Основная идея метода:
В качестве функции сравнения 2 - моделей используем функцию cosine-distance.
Предварительно на трейн данных обучаем матрицу преобразования whitened PCA.

Итого:
score(x,y) = (Wx)_T * (Wy) / (norm(Wx) * norm(Wy)), где W - whitened PCA matrix.


fr_fa.png - график fr/fa
hist.png - гистограммы скоров

EER = 3.3%
