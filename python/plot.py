import matplotlib.pyplot as plt
import pandas as pd

column_labs = ["batch_num", "acc"]
df = pd.read_csv("./logs/log1/acc.csv", names=column_labs)

fig, ax1 = plt.subplots()
ax1.plot(df["batch_num"], df["acc"], color="red", label="Accuracy")
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Batch")

column_labs2 = ["batch_num", "loss"]
df = pd.read_csv("./logs/log1/loss.csv", names=column_labs2)

fig2, ax2 = plt.subplots()
ax2.plot(df["batch_num"], df["loss"], color="red", label="Loss")
ax2.set_ylabel("Loss")
ax2.set_xlabel("Batch")

plt.show()
