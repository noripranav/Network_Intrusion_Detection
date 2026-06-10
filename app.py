from flask import Flask, render_template, request, jsonify
import subprocess
import threading
import os
import sys
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Shared prediction status
prediction_status = {
    'status': 'idle',        # idle, processing, done
    'predictions': None
}

def handle_capture_and_predict():
    try:
        print("[*] Running capture_script.py (requires sudo)...")
        subprocess.run(['sudo', sys.executable, 'capture_script.py'], check=True)

        print("[*] Capture complete. Running prediction...")
        subprocess.run([sys.executable, 'predict1.py'], check=True)

        print("[✓] Prediction complete. Reading predictions...")
        with open('predictions.csv', 'r') as f:
            lines = f.readlines()[1:]
            predictions = [line.strip() for line in lines]

        prediction_status['predictions'] = predictions
        prediction_status['status'] = 'done'

    except subprocess.CalledProcessError as e:
        print(f"[✗] Error during execution: {e}")
        prediction_status['status'] = 'error'
        prediction_status['predictions'] = [str(e)]
    
    except Exception as e:
        print(f"[✗] General error: {e}")
        prediction_status['status'] = 'error'
        prediction_status['predictions'] = [str(e)]


@app.route('/')
def index():
    return render_template('index.html')

latest_output = ""

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    global latest_output
    try:
        if prediction_status['status'] == 'processing':
            return jsonify({'status': 'Already processing...'})

        # Reset prediction state
        prediction_status['status'] = 'processing'
        prediction_status['predictions'] = None

        # Start the background capture + predict
        thread = threading.Thread(target=handle_capture_and_predict)
        thread.start()

        return jsonify({'status': 'Monitoring started...'})

    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'error': e.stderr}), 500

@app.route('/get-predictions', methods=['GET'])
def get_predictions():
    if prediction_status['status'] == 'done':
        if os.path.exists("prediction_summary.txt"):
            with open("prediction_summary.txt", "r") as f:
                summary_text = f.read()

            # Optional: reset status so old results are cleared next time
            prediction_status['status'] = 'idle'
            prediction_status['predictions'] = None

            return jsonify({
                'status': 'done',
                'output': summary_text,
                'predictions': prediction_status['predictions']
            })

        else:
            return jsonify({'status': 'done', 'output': 'No summary file found.'})

    elif prediction_status['status'] == 'processing':
        return jsonify({'status': 'processing'})
    else:
        return jsonify({'status': 'idle', 'output': 'Waiting to start prediction...'})

# @app.route('/get-predictions', methods=['GET'])
# def get_predictions():
#     status = prediction_status['status']
    
#     if status == 'done':
#         result = prediction_status['predictions']
#         # Reset after serving
#         prediction_status['status'] = 'idle'
#         prediction_status['predictions'] = None
#         return jsonify({'predictions': result})
#     elif status == 'processing':
#         return jsonify({'status': 'processing'})
#     elif status == 'error':
#         return jsonify({'error': prediction_status['predictions']})
#     else:
#         return jsonify({'status': 'idle', 'message': 'Click capture to start monitoring.'})

    
@app.route('/manual-check', methods=['POST'])
def manual_check():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'})

        filename = secure_filename(file.filename)
        file.save(filename)

        print(f"[*] Received file: {filename}")

        # Modify this part based on your actual processing logic
        # Assuming predict1.py can take a file argument for manual check:
        subprocess.run(['python', 'predict1.py', '--input', filename], check=True)

        # Read prediction result from predictions.csv as before
        with open('predictions.csv', 'r') as f:
            lines = f.readlines()[1:]
            predictions = [line.strip() for line in lines]

        summary_text = ""
        summary_path = os.path.join('prediction_summary.txt')
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary_text = f.read()

        # ⬇️ Include summary in response
        return jsonify({
            'predictions': predictions,
            'summary': summary_text
        })

    except subprocess.CalledProcessError as e:
        print(f"[✗] Prediction error: {e}")
        return jsonify({'error': str(e)})
    except Exception as e:
        print(f"[✗] General error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render provides PORT as env variable
    app.run(host='0.0.0.0', port=port)

