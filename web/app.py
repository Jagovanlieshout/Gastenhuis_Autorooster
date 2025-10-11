import os
import tempfile
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
import csv
import traceback


from app import preprocess_data, auto_rooster, validate_auto_rooster

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB upload limit
app.template_folder = os.path.join(os.path.dirname(__file__), "templates")

ALLOWED_EXT = {"csv", "xlsx"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def read_dataframe(file_storage):
    content = file_storage.read()
    file_storage.seek(0)
    try:
        dialect = csv.Sniffer().sniff(content.decode('utf-8'))
        sep = dialect.delimiter
    except Exception:
        sep = ','

    if file_storage.filename.lower().endswith("xlsx"):
        return pd.read_excel(file_storage)
    return pd.read_csv(file_storage, sep=sep)

def read_workers(file_storage):
    return pd.read_excel(file_storage, sheet_name='Tabellen', usecols='T:AE', skiprows=1)

def read_vast_rooster(file_storage):
    return pd.read_excel(file_storage, sheet_name='Vaste roosters', usecols='A:E', skiprows=1)

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/schedule", methods=["POST"])
def generate_schedule():
    try:
        # --- 1. Get uploaded files ---
        #print("=== LOADING FILES ===")
        required_files = ["workers", "onb", "prev_assignments"]
        for rf in required_files:
            if rf not in request.files:
                return jsonify({"error": f"Missing required file: {rf}"}), 400

        #workers_df = read_dataframe(request.files["workers"])\
        workers_df = read_workers(request.files["workers"])
        vast_rooster_df = read_vast_rooster(request.files["workers"])
        onb_df = read_dataframe(request.files["onb"])
        # if no prev_assignments provided, set to None
        if request.files["prev_assignments"].filename == "":
            prev_df = None
        else:    
            prev_df = read_dataframe(request.files["prev_assignments"])

        #print(workers_df.head())
        
        # --- 2. Preprocess & solve ---
        #print("=== LOADING & PREPROCESSING DATA ===")
        data = preprocess_data(workers_df, onb_df, prev_df, vast_rooster_df)
    
        #print("=== SOLVING SCHEDULE ===")
        result = auto_rooster(data, time_limit_s=60)

        assignments_df = result["assignments_df"]

        # --- 3. Validate schedule ---
        errors = validate_auto_rooster(data, result)
        if errors:
            return jsonify({"status": "failed", "validation_errors": errors}), 400

        # --- 4. Save CSV to temp file ---
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        assignments_df.to_csv(tmpfile.name, index=False)
        tmpfile.flush()
    
        # --- 5. Return stats as JSON ---
        stats = {
            "start_date": str(assignments_df["shift_date"].min()),
            "end_date": str(assignments_df["shift_date"].max()),
            "total_shifts": len(data["shifts"]),
            "num_employees": assignments_df["employee_id"].nunique(),
            "shifts_filled": int(assignments_df["shift_filled"].sum()),
            "shifts_unfilled": len(data["shifts"]) - int(assignments_df["shift_filled"].sum()),
            "download_url": f"/download/{os.path.basename(tmpfile.name)}"
        }
    

        return jsonify({"status": "success", "stats": stats})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e),
                        "trace": traceback.format_exc()}), 500

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    return send_file(os.path.join(tempfile.gettempdir(), filename),
                     as_attachment=True,
                     download_name='Rooster.csv')  # Force download with a generic name
    


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
