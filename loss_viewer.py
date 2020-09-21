"""
    loss.csv の内容を描画
    第一列が生成器、第二列が鑑別器の損失関数の値
"""

import csv
import matplotlib.pyplot as plt

epochs = []
g_loss, d_loss = [], []  ## eporch毎の生成器と鑑別器の損失関数の値がそれぞれ格納
with open('loss.csv', 'r') as f:
    reader = csv.reader(f)
    epoch = 0
    for row in reader:
        epochs.append(epoch)
        g_loss_row = float(row[0])
        d_loss_row = float(row[1])
        g_loss.append(g_loss_row)
        d_loss.append(d_loss_row)
        epoch += 1

## Generatorの損失関数値
plt.title("Generator, Loss-Epochs Plot")
plt.xlabel("Epochs")
plt.ylabel("Value of Loss Function")
plt.plot(epochs, g_loss)
plt.show()
plt.close()

## Discriminatorの損失関数値
plt.title("Discriminator, Loss-Epochs Plot")
plt.xlabel("Epochs")
plt.ylabel("Value of Loss Function")
plt.plot(epochs, d_loss)
plt.show()
plt.close()