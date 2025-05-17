# api/wdfidf.py
import math
import requests
from bs4 import BeautifulSoup
from collections import Counter
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/wdfidf', methods=['POST'])
def wdfidf():
    try:
        payload = request.get_json()
        urls    = payload.get("urls", [])
        top_n   = payload.get("top_n", 10)

        docs = []
        for url in urls:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return jsonify(error=f"Fehler beim Abruf: {url}"), 400
            text = BeautifulSoup(r.text, "html.parser").get_text(separator=" ")
            docs.append(text)

        counters = [Counter(d.lower().split()) for d in docs]
        N = len(counters)
        scores = {}
        for term in set().union(*[c.keys() for c in counters]):
            df  = sum(1 for c in counters if term in c)
            idf = math.log((N + 1)/(df + 1)) + 1
            for c in counters:
                tf = c[term]
                total = sum(c.values()) or 1
                wdf = tf/total
                scores[term] = scores.get(term, 0) + wdf*idf

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        result = [{"term": t, "score": s} for t, s in top]

        return jsonify(terms=result)

    except Exception as e:
        return jsonify(error=str(e)), 500
