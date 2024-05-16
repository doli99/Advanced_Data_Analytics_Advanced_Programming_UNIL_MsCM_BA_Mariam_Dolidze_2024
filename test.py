import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import seaborn as sns
import threading

# Ensure Matplotlib backend is set to TkAgg
plt.switch_backend('TkAgg')

def test_script():
    root = tk.Tk()
    root.title("Test Taxi Trip EDA")

    # Load a sample dataframe
    df = pd.DataFrame({
        'fare_amount': [10, 20, 30],
        'trip_distance': [1.0, 2.5, 3.0]
    })

    # Simple visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['fare_amount'], kde=True, color='#FFD700', edgecolor='#000000', ax=ax)
    ax.set_title('Histogram of fare_amount', fontsize=16)
    ax.set_xlabel('fare_amount', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)

    # Embed the plot in the Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    test_script()
