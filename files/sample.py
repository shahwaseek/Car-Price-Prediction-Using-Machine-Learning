import pandas as pd
import numpy as np
from tkinter import Tk, Label, Entry, Button, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # For saving the model


class CarPricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Price Predictor")
        self.root.geometry("600x500")
        self.data = None
        self.model = None

        # Project Heading
        Label(root, text="Car Price Prediction System", font=("Helvetica", 20, "bold"), fg="blue").pack(pady=20)

        # Buttons and Features
        Button(root, text="Load Dataset", command=self.load_dataset, width=20).pack(pady=10)
        Button(root, text="Train Model", command=self.train_model, width=20).pack(pady=10)
        Button(root, text="Evaluate Model", command=self.evaluate_model, width=20).pack(pady=10)

        Label(root, text="Input Features for Prediction", font=("Helvetica", 14)).pack(pady=10)
        self.input_fields = ["Year", "KM Driven", "Fuel (0/1/2)", "Seller Type (0/1)",
                             "Transmission (0/1)", "Owner (0/1/2)"]
        self.entries = []
        for field in self.input_fields:
            Label(root, text=field).pack()
            entry = Entry(root)
            entry.pack()
            self.entries.append(entry)

        Button(root, text="Predict Price", command=self.predict_price, width=20).pack(pady=10)
        Button(root, text="Save Model", command=self.save_model, width=20).pack(pady=10)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("Success", "Dataset Loaded Successfully")
            print("Dataset Preview:")
            print(self.data.head())

    def preprocess_data(self):
        if self.data is None:
            messagebox.showerror("Error", "No dataset loaded!")
            return None, None

        # Drop irrelevant columns
        data = self.data.drop(columns=['name'], axis=1, errors='ignore')

        # Handle missing values
        data = data.dropna()

        # Encode categorical variables
        label_encoder = LabelEncoder()
        for col in ['fuel', 'seller_type', 'transmission', 'owner']:
            if col in data.columns:
                data[col] = label_encoder.fit_transform(data[col])

        # Features and target
        X = data.drop('selling_price', axis=1)
        y = data['selling_price']
        return X, y

    def train_model(self):
        if self.data is None:
            messagebox.showerror("Error", "No dataset loaded!")
            return

        X, y = self.preprocess_data()
        if X is None or y is None:
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.X_test, self.y_test = X_test, y_test
        messagebox.showinfo("Success", "Model Trained Successfully!")

    def evaluate_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not trained!")
            return

        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)

        eval_message = (f"Evaluation Metrics:\n"
                        f"MAE: {mae:.2f}\n"
                        f"MSE: {mse:.2f}\n"
                        f"RMSE: {rmse:.2f}\n"
                        f"R2 Score: {r2:.2f}")
        messagebox.showinfo("Evaluation", eval_message)

    def predict_price(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not trained!")
            return

        try:
            # Gather inputs from user entries
            input_values = [float(entry.get()) for entry in self.entries]

            # Predict the price
            prediction = self.model.predict([input_values])[0]

            # Format the price with INR symbol and commas
            formatted_price = f"₹{prediction:,.2f}"

            # Show the predicted price in a message box
            messagebox.showinfo("Prediction", f"Predicted Price: {formatted_price}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input! Please enter valid data.\n\n{e}")

    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not trained!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        if file_path:
            joblib.dump(self.model, file_path)
            messagebox.showinfo("Success", "Model Saved Successfully!")


# Main
if __name__ == '__main__':
    root = Tk()
    app = CarPricePredictorApp(root)
    root.mainloop()
