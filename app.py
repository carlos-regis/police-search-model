from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/should_search', methods=['POST'])
def should_search():
    outcome = False
    return jsonify({
        'outcome': outcome
    })


if __name__ == "__main__":
    app.run(debug=True)
