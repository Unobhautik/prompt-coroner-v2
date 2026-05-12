"""
app.py — Flask server for Prompt Coroner v2
Thin wrapper around the LangGraph pipeline.
"""
import os
from flask import Flask, request, jsonify, send_from_directory
from graph import run_autopsy

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/autopsy", methods=["POST"])
def autopsy():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    output = data.get("output", "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = run_autopsy(prompt, output)
        return jsonify({
            "health_score":    result.get("health_score", 0),
            "severity":        result.get("severity", "medium"),
            "death_cause":     result.get("death_cause", ""),
            "failure_tags":    result.get("failure_tags", []),
            "autopsy_rows":    result.get("autopsy_rows", []),
            "similar_cases":   result.get("similar_cases", []),
            "deep_dive_notes": result.get("deep_dive_notes", ""),
            "reformulations":  result.get("reformulations", []),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        print("\n❌  Set GROQ_API_KEY before running.\n")
        exit(1)
    print("\n🔬 Prompt Coroner v2 — LangGraph Edition")
    print("👉 Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)
