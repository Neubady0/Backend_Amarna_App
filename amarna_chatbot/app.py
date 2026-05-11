from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    # Placeholder for the Amarna Chatbot logic
    data = request.get_json()
    message = data.get('message', '')
    
    return jsonify({
        'reply': f'Echo from Amarna Chatbot: {message}',
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
