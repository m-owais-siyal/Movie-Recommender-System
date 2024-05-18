import tkinter as tk
from tkinter import ttk
import pandas as pd
import matrix_factorization_rs as mx
import content_based_rs as cb_rs
import naive_rs

# Function to mimic placeholder functionality
def set_placeholder(entry, placeholder_text):
    entry.insert(0, placeholder_text)
    entry.config(fg='grey')

    def on_focus_in(event):
        if entry.get() == placeholder_text:
            entry.delete(0, 'end')
            entry.config(fg='black')

    def on_focus_out(event):
        if not entry.get():
            entry.insert(0, placeholder_text)
            entry.config(fg='grey')

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

# Initialize main application
rs_project = tk.Tk()
rs_project.title("Movie Recommender System")
rs_project.geometry("700x600")
rs_project.configure(bg="#2b2b2b")

# Function to load and display data from CSV files
def display_csv_data(file_name):
    try:
        df = pd.read_csv(file_name)
        sample_data.delete(1.0, tk.END)
        sample_data.insert(tk.END, f"Data Head from {file_name}:\n\n")
        sample_data.insert(tk.END, df.head().to_string(index=False))
    except Exception as e:
        sample_data.delete(1.0, tk.END)
        sample_data.insert(tk.END, f"Error reading file {file_name}: {e}")

# Function to fetch recommendations based on the selected algorithm
def fetch_recommendations():
    user_input = user_entry.get()
    algorithm = algorithm_var.get()

    recommendations = []
    accuracy = None

    if algorithm == "Matrix Factorization RS":
        recommendations = mx.matrix_factorization_rs(user_input)
    elif algorithm == "Content-Based RS":
        recommendations = cb_rs.content_based_filtering(user_input)
    elif algorithm == "Naive-Based RS":
        recommendations, accuracy = naive_rs.naive_bayes_cf(user_input)

    recommendation_display.delete("1.0", tk.END)
    if accuracy is not None:
        recommendation_display.insert(tk.END, f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
    for rec in recommendations:
        recommendation_display.insert(tk.END, rec + '\n')

# Create a Frame to hold the content
main_frame = ttk.Frame(rs_project, padding="10 10 10 10")
main_frame.grid(row=0, column=0, sticky="nsew")

# Style configuration
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12), background="#2b2b2b", foreground="#ffffff")
style.configure("TButton", font=("Arial", 12, "bold"), foreground="black", background="#5b5b5b")
style.configure("TFrame", background="#2b2b2b")
style.configure("TCombobox", font=("Arial", 12))

# Buttons to load each dataset
button_movies = ttk.Button(main_frame, text="Load Movies Data", command=lambda: display_csv_data("movies.csv"))
button_ratings = ttk.Button(main_frame, text="Load Ratings Data", command=lambda: display_csv_data("ratings.csv"))
button_tags = ttk.Button(main_frame, text="Load Tags Data", command=lambda: display_csv_data("tags.csv"))

# Layout for the dataset buttons
button_movies.grid(row=0, column=0, padx=10, pady=10)
button_ratings.grid(row=0, column=1, padx=10, pady=10)
button_tags.grid(row=0, column=2, padx=10, pady=10)

# Text widget to display the dataset contents
sample_data = tk.Text(main_frame, height=7, width=80, font=("Arial", 10))
sample_data.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# Recommendation Algorithm Selection
algorithm_var = tk.StringVar(value="Matrix Factorization RS")
algorithm_label = ttk.Label(main_frame, text="Select Recommendation Algorithm")
algorithm_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
algorithm_combobox = ttk.Combobox(main_frame, textvariable=algorithm_var, values=["Matrix Factorization RS", "Content-Based RS", "Naive-Based RS"], state="readonly")
algorithm_combobox.grid(row=2, column=1, padx=10, pady=10)

# Entry to take user input
user_entry = tk.Entry(main_frame, width=30, font=("Arial", 10))
user_entry.grid(row=3, column=0, padx=10, pady=10)
set_placeholder(user_entry, "Enter user id here")

# Submit button to fetch recommendations
submit_button = ttk.Button(main_frame, text="Get Recommendations", command=fetch_recommendations)
submit_button.grid(row=3, column=1, padx=10, pady=10)

# Text widget to display the recommendations
recommendation_display = tk.Text(main_frame, height=7, width=80, font=("Arial", 10))
recommendation_display.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

# Start the Tkinter main loop
tk.mainloop()
