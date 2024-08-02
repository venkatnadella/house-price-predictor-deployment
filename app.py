from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flashing messages
model = load('house_price_model.joblib')

# Define the expected columns
expected_columns = ['Size (sq ft)', 'Bedrooms', 'Age (years)']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or not request.files['file'].filename:
        flash('Please select a file before uploading.', 'error')
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    if file:
        data = pd.read_csv(file)
        return process_data(data)

@app.route('/manual_entry', methods=['POST'])
def manual_entry():
    data_string = request.form.get('data', '').strip()
    
    if not data_string:
        flash('Please enter data before submitting.', 'error')
        return redirect(url_for('home'))

    # Convert the data string into a list of lists, skipping the first row (header)
    data_lines = data_string.split('\n')
    header = data_lines[0].split(',')
    data_rows = [line.split(',') for line in data_lines[1:] if line.strip()]

    # Create DataFrame
    data = pd.DataFrame(data_rows, columns=expected_columns)
    
    # Convert columns to appropriate types
    try:
        data = data.astype(float)
    except ValueError:
        flash('Data contains invalid values. Please check your input.', 'error')
        return redirect(url_for('home'))
    
    return process_data(data)


def process_data(data):
    # Ensure the data has the expected columns
    if not all(col in data.columns for col in expected_columns):
        return "Data does not have the expected columns!", 400

    X = data[expected_columns].astype(float)

    y_pred = model.predict(X)

    fig, ax = plt.subplots()

    if len(y_pred) == 1:
        ax.bar(['Predicted Price'], y_pred)
        ax.set_ylim(0, max(y_pred) * 1.2)
    else:
        ax.plot(y_pred, label='Predicted Price', marker='o')

    for i, txt in enumerate(y_pred):
        ax.annotate(f'{txt:.2f}', (i, y_pred[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_title('Predicted House Prices')
    ax.set_xlabel('Record Index')
    ax.set_ylabel('Predicted Price')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('ascii')

    return render_template('results.html', img_data=img)

if __name__ == '__main__':
    app.run(debug=True)
