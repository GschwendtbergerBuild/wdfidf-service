# api/chat.py
import os
import json
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
# Dein API-Key wird per ENV-Variable gezogen
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Du bist SEO Master Bot – ein smarter, strategischer und motivierender SEO-Coach auf Profi-Level.
Nutze bei Bedarf die Funktion wdfidf_analysis, um echte WDF*IDF-Analysen durchzuführen."""

# Definition der Function (muss identisch zur Dashboard-Funktion sein)
WDFIDF_FUNCTION = {
    "name": "wdfidf_analysis",
    "description": "Führt eine WDF*IDF-Analyse für gegebene URLs durch und liefert die wichtigsten Terme mit Scores.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {"type": "array", "description": "Liste von Seiten-URLs zur Analyse (eigene Seite + Konkurrenz).", "items": {"type": "string"}},
            "top_n": {"type": "integer", "description": "Anzahl der Top-Terme (optional, Standard 10)."}
        },
        "required": ["urls"]
    }
}

def call_wdfidf_service(urls, top_n=10):
    """Wrapper: ruft deine WDF*IDF-API auf."""
    base_url = os.getenv("WDFIDF_BASE_URL", "http://localhost:3000")
    res = requests.post(
        f"{base_url}/api/wdfidf",
        json={"urls": urls, "top_n": top_n},
        timeout=15
    )
    res.raise_for_status()
    return res.json()  # { "terms": [ { "term": "...", "score": ... }, ... ] }

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_input = data.get("message", "")

    # 1) Erstaufruf mit Function-Definition
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        functions=[WDFIDF_FUNCTION],
        function_call="auto"
    )

    message = resp.choices[0].message

    # 2) Check auf Funktionsaufruf
    if message.function_call and message.function_call.name == "wdfidf_analysis":
        raw_args = message.function_call.arguments
        params = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        urls = params.get("urls", [])
        top_n = params.get("top_n", 10)

        print("▶️ WDF*IDF call params:", params)

        try:
            result = call_wdfidf_service(urls, top_n)
        except Exception as e:
            return jsonify({"role": "assistant", "content": f"⚠️ Fehler bei der WDF*IDF-Analyse: {e}"})

        # 3) Follow-up an OpenAI
        followup = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": None, "function_call": {"name": "wdfidf_analysis", "arguments": json.dumps(params)}},
                {"role": "function", "name": "wdfidf_analysis", "content": json.dumps(result)}
            ]
        )
        # 4) Nachricht extrahieren und als JSON zurückgeben
        msg2 = followup.choices[0].message
        return jsonify({"role": msg2.role, "content": msg2.content})

    # 5) Kein Funktionsaufruf → normale Bot-Antwort
    return jsonify({"role": message.role, "content": message.content})

if __name__ == "__main__":
    app.run(port=3000, debug=True)
