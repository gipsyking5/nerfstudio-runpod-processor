from flask import Flask, request, jsonify
import os
import subprocess # To run command-line tools like Nerfstudio
import firebase_admin
from firebase_admin import credentials, firestore, storage
# import ffmpeg # Uncomment if using ffmpeg-python for frame extraction
# import cv2 # Uncomment if using OpenCV for frame extraction
import shutil # For file/directory management
import time
import uuid # For unique temporary folder names
import json # To parse the JSON key from environment variable

# --- Configuration ---
# SECURELY Initialize Firebase Admin SDK using RunPod Environment Variables

# 1. Get the Service Account Key JSON string from the environment variable
SERVICE_ACCOUNT_JSON_STRING = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

# 2. Get the Storage Bucket Name from the environment variable
# !!! User needs to set FIREBASE_STORAGE_BUCKET in RunPod !!!
STORAGE_BUCKET_NAME = os.environ.get("FIREBASE_STORAGE_BUCKET")

db = None
bucket = None

try:
    if not SERVICE_ACCOUNT_JSON_STRING:
        print(f"CRITICAL ERROR: Environment variable FIREBASE_SERVICE_ACCOUNT_JSON not found. Cannot initialize Firebase.")
        raise ValueError("Environment variable FIREBASE_SERVICE_ACCOUNT_JSON is not set.")

    if not STORAGE_BUCKET_NAME:
        print(f"CRITICAL ERROR: Environment variable FIREBASE_STORAGE_BUCKET not found. Cannot initialize Firebase.")
        raise ValueError("Environment variable FIREBASE_STORAGE_BUCKET is not set.")

    # Parse the JSON string from the environment variable
    service_account_info = json.loads(SERVICE_ACCOUNT_JSON_STRING)
    cred = credentials.Certificate(service_account_info)

    # Initialize Firebase with the bucket name from the environment variable
    firebase_admin.initialize_app(cred, {
       'storageBucket': STORAGE_BUCKET_NAME # Use the variable here
    })
    db = firestore.client()
    bucket = storage.bucket() # Get bucket instance after initialization
    print(f"Firebase Admin SDK initialized successfully using Environment Variables. Target bucket: {STORAGE_BUCKET_NAME}")

except Exception as e:
    print(f"CRITICAL: Failed to initialize Firebase Admin SDK: {e}")
    # The app likely cannot function without Firebase, so errors here are serious.
# --- End Firebase Init ---


app = Flask(__name__)

# --- Helper Functions ---
# These functions now rely on 'bucket' being correctly initialized above.
def download_video(storage_path, local_path):
    """Downloads video from Firebase Storage."""
    if not bucket:
        raise Exception("Firebase Storage bucket not initialized. Cannot download.")
    blob = bucket.blob(storage_path)
    print(f"Attempting to download gs://{bucket.name}/{storage_path} to {local_path}...")
    blob.download_to_filename(local_path)
    print("Download complete.")

def upload_splat(local_ply_path, storage_path):
    """Uploads the resulting PLY file to Firebase Storage."""
    if not bucket:
        raise Exception("Firebase Storage bucket not initialized. Cannot upload.")
    blob = bucket.blob(storage_path)
    print(f"Attempting to upload {local_ply_path} to gs://{bucket.name}/{storage_path}...")
    blob.upload_from_filename(local_ply_path)
    # Make the file publicly readable - adjust if you need signed URLs instead
    blob.make_public()
    print(f"Upload complete. Public URL: {blob.public_url}")
    return blob.public_url

def update_firestore(doc_id, status, splat_url=None):
    """Updates the Firestore document status and URL."""
    if not db:
        raise Exception("Firestore client not initialized. Cannot update status.")
    # !!! IMPORTANT: Make sure 'discoveries' is your correct collection name !!!
    doc_ref = db.collection('discoveries').document(doc_id)
    update_data = {'processingStatus': status}
    if splat_url:
        update_data['splatModelUrl'] = splat_url
    print(f"Updating Firestore doc '{doc_id}' with: {update_data}")
    doc_ref.update(update_data)
    print("Firestore update complete.")

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    return jsonify({"status": "healthy", "firebase_initialized": bool(db and bucket)}), 200

@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    """
    Main endpoint to process video: download, run Nerfstudio, upload result, update DB.
    Expects JSON: {"videoStoragePath": "path/in/storage.webm", "firestoreDocId": "document_id", "userId": "user_id"}
    """
    if not (db and bucket):
         # If Firebase didn't initialize, fail early.
         print("Error: Firebase not initialized. Check environment variables and key.")
         return jsonify({"error": "Backend not fully initialized (Firebase connection missing)."}), 503 # Service Unavailable

    try:
        data = request.get_json()
        if not data or 'videoStoragePath' not in data or 'firestoreDocId' not in data:
            return jsonify({"error": "Missing 'videoStoragePath' or 'firestoreDocId' in JSON payload"}), 400

        video_storage_path = data['videoStoragePath']
        doc_id = data['firestoreDocId']
        user_id = data.get('userId', 'unknown_user') # Get user ID for paths, default if missing

        # --- Processing Steps ---
        job_id = str(uuid.uuid4())
        base_work_dir = "/tmp/nerfstudio_jobs"
        job_dir = os.path.join(base_work_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        print(f"[{job_id}] Created temporary job directory: {job_dir}")

        local_video_path = os.path.join(job_dir, os.path.basename(video_storage_path))
        data_dir = os.path.join(job_dir, "nerf_data") # Input for ns-train
        output_dir = os.path.join(job_dir, "nerf_output") # Output for ns-train/ns-export

        try:
            # 2. Download the video from Firebase Storage
            print(f"[{job_id}] Downloading video: {video_storage_path}")
            start_time = time.time()
            download_video(video_storage_path, local_video_path)
            print(f"[{job_id}] Video downloaded in {time.time() - start_time:.2f} seconds.")

            # 3. Process Video with Nerfstudio (COMMANDS NEED VERIFICATION/ADJUSTMENT)
            print(f"[{job_id}] --- Starting Nerfstudio Processing ---")
            start_time = time.time()

            # --- === NERFSTUDIO COMMANDS (VERIFY & ADJUST) === ---
            process_cmd = [
                "ns-process-data", "video",
                "--data", local_video_path,
                "--output-dir", data_dir,
                "--verbose"
                # Add tuning params as needed
            ]
            print(f"[{job_id}] Running command: {' '.join(process_cmd)}")
            subprocess.run(process_cmd, check=True)
            print(f"[{job_id}] ns-process-data finished.")

            train_cmd = [
                "ns-train", "splatfacto",
                "--data", data_dir,
                "--output-dir", output_dir,
                "--vis", "none",
                "--max-num-iterations", "5000", # Adjust
                "--pipeline.model.sh-degree", "0", # Adjust
            ]
            print(f"[{job_id}] Running command: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True)
            print(f"[{job_id}] ns-train splatfacto finished.")

            # Find config.yml
            splatfacto_output_dir = os.path.join(output_dir, "splatfacto")
            try:
                training_dirs = [d for d in os.listdir(splatfacto_output_dir) if os.path.isdir(os.path.join(splatfacto_output_dir, d))]
                training_dirs.sort()
                latest_training_dir = os.path.join(splatfacto_output_dir, training_dirs[-1])
                config_path = os.path.join(latest_training_dir, "config.yml")
                if not os.path.exists(config_path): raise FileNotFoundError("config.yml missing")
            except Exception as find_err:
                 print(f"[{job_id}] Error finding config.yml automatically: {find_err}")
                 raise FileNotFoundError(f"Could not find config.yml in {splatfacto_output_dir}")

            ply_output_filename = "splat.ply"
            ply_output_path = os.path.join(output_dir, ply_output_filename)

            export_cmd = [
                "ns-export", "gaussian-splat",
                "--load-config", config_path,
                "--output-dir", output_dir,
                "--output-name", ply_output_filename
            ]
            print(f"[{job_id}] Running command: {' '.join(export_cmd)}")
            subprocess.run(export_cmd, check=True)
            print(f"[{job_id}] ns-export gaussian-splat finished.")

            if not os.path.exists(ply_output_path):
                 raise FileNotFoundError(f"Output PLY file not found after export at: {ply_output_path}")
            # --- === END NERFSTUDIO COMMANDS === ---

            total_processing_time = time.time() - start_time
            print(f"[{job_id}] --- Nerfstudio processing finished in {total_processing_time:.2f} seconds. ---")

            # 4. Upload the resulting PLY file to Firebase Storage
            print(f"[{job_id}] Uploading result: {ply_output_path}")
            start_time = time.time()
            splat_storage_path = f"splat-models/{user_id}/{doc_id}.ply"
            public_url = upload_splat(ply_output_path, splat_storage_path)
            print(f"[{job_id}] Result uploaded in {time.time() - start_time:.2f} seconds.")

            # 5. Update Firestore status to 'complete' and add the URL
            update_firestore(doc_id, 'complete', public_url)

            # 6. Clean up temporary files ONLY on success
            print(f"[{job_id}] Cleaning up job directory: {job_dir}")
            shutil.rmtree(job_dir)

            return jsonify({"status": "processing complete", "splatUrl": public_url}), 200

        # --- Error Handling during Processing Steps ---
        except subprocess.CalledProcessError as e:
            error_message = f"Nerfstudio command failed (Code {e.returncode}). Check RunPod logs."
            print(f"[{job_id}] {error_message}")
            print(f"[{job_id}] Stderr: {e.stderr}") # Log stderr for debugging
            try:
                update_firestore(doc_id, 'failed')
            except Exception as db_err:
                print(f"[{job_id}] Also failed to update Firestore status to failed: {db_err}")
            print(f"[{job_id}] Cleaning up failed job directory: {job_dir}")
            if os.path.exists(job_dir): shutil.rmtree(job_dir)
            return jsonify({"error": "Nerfstudio processing failed", "details": error_message}), 500

        except Exception as e:
            error_message = f"An error occurred during processing: {str(e)}"
            print(f"[{job_id}] {error_message}")
            try:
                update_firestore(doc_id, 'failed')
            except Exception as db_err:
                print(f"[{job_id}] Also failed to update Firestore status to failed: {db_err}")
            print(f"[{job_id}] Cleaning up failed job directory: {job_dir}")
            if os.path.exists(job_dir): shutil.rmtree(job_dir)
            return jsonify({"error": "An internal server error occurred during processing", "details": error_message}), 500

    # --- Error Handling for Request Issues (e.g., bad JSON) ---
    except Exception as e:
        error_message = f"Error handling request: {str(e)}"
        print(f"[PRE-JOB ERROR] {error_message}")
        return jsonify({"error": "Failed to handle request", "details": error_message}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to be accessible outside the container
    port = int(os.environ.get("PORT", os.environ.get("RUNPOD_REALTIME_PORT", 8080)))
    print(f"Starting Flask server on host 0.0.0.0 port {port}")
    # Use Flask's built-in server for simplicity with RunPod Serverless (1 worker)
    app.run(host='0.0.0.0', port=port, debug=False)
