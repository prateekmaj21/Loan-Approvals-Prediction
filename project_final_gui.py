import tkinter as tk
from tkinter import messagebox
from pickle import load
import numpy as np

# Load the model and scaler
model = load(open('model.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))

# Function to perform prediction
def predict_loan_eligibility():
    try:
        # Get user inputs
        no_of_dependents = int(dependents_entry.get())
        education = education_var.get()  # Use directly as 0 or 1
        self_employed = self_employed_var.get()  # Use directly as 0 or 1
        income_annum = float(income_entry.get())
        loan_amount = float(loan_amount_entry.get())
        loan_term = int(loan_term_entry.get())
        cibil_score = int(cibil_score_entry.get())
        residential_assets_value = float(residential_assets_entry.get())
        commercial_assets_value = float(commercial_assets_entry.get())
        luxury_assets_value = float(luxury_assets_entry.get())
        bank_asset_value = float(bank_asset_entry.get())
        
        # Prepare the input array
        sample = np.array([[no_of_dependents, education, self_employed, income_annum,
                            loan_amount, loan_term, cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value, bank_asset_value]])
        
        # Scale the input
        sample_scaled = scaler.transform(sample)
        
        # Predict using the model
        prediction = model.predict(sample_scaled)
        
        # Display the prediction
        if prediction[0] == 1:
            result_label.config(text="Approved", fg="green")
        else:
            result_label.config(text="Rejected", fg="red")
    
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {str(e)}")

# Create the main window
root = tk.Tk()
root.title("Loan Eligibility Predictor")

# Create labels and entries for each input variable
tk.Label(root, text="Number of Dependents:").grid(row=0, column=0, sticky="w")
dependents_entry = tk.Entry(root)
dependents_entry.grid(row=0, column=1)

tk.Label(root, text="Education:").grid(row=1, column=0, sticky="w")
education_var = tk.IntVar()
tk.Radiobutton(root, text="Graduate", variable=education_var, value=1).grid(row=1, column=1, sticky="w")
tk.Radiobutton(root, text="Not Graduate", variable=education_var, value=0).grid(row=1, column=2, sticky="w")

tk.Label(root, text="Self Employed:").grid(row=2, column=0, sticky="w")
self_employed_var = tk.IntVar()
tk.Radiobutton(root, text="Yes", variable=self_employed_var, value=1).grid(row=2, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=self_employed_var, value=0).grid(row=2, column=2, sticky="w")

tk.Label(root, text="Annual Income:").grid(row=3, column=0, sticky="w")
income_entry = tk.Entry(root)
income_entry.grid(row=3, column=1)

tk.Label(root, text="Loan Amount:").grid(row=4, column=0, sticky="w")
loan_amount_entry = tk.Entry(root)
loan_amount_entry.grid(row=4, column=1)

tk.Label(root, text="Loan Term :").grid(row=5, column=0, sticky="w")
loan_term_entry = tk.Entry(root)
loan_term_entry.grid(row=5, column=1)

tk.Label(root, text="CIBIL Score:").grid(row=6, column=0, sticky="w")
cibil_score_entry = tk.Entry(root)
cibil_score_entry.grid(row=6, column=1)

tk.Label(root, text="Value of Residential Assets:").grid(row=7, column=0, sticky="w")
residential_assets_entry = tk.Entry(root)
residential_assets_entry.grid(row=7, column=1)

tk.Label(root, text="Value of Commercial Assets:").grid(row=8, column=0, sticky="w")
commercial_assets_entry = tk.Entry(root)
commercial_assets_entry.grid(row=8, column=1)

tk.Label(root, text="Value of Luxury Assets:").grid(row=9, column=0, sticky="w")
luxury_assets_entry = tk.Entry(root)
luxury_assets_entry.grid(row=9, column=1)

tk.Label(root, text="Value of Bank Assets:").grid(row=10, column=0, sticky="w")
bank_asset_entry = tk.Entry(root)
bank_asset_entry.grid(row=10, column=1)

# Button to predict loan eligibility
predict_button = tk.Button(root, text="Predict", command=predict_loan_eligibility)
predict_button.grid(row=11, column=0, columnspan=2, pady=10)

# Label to display result
result_label = tk.Label(root, text="", font=("Helvetica", 18))
result_label.grid(row=12, column=0, columnspan=2)

# Start the main loop
root.mainloop()
