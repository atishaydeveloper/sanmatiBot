import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from twilio_chatbot import process_message, reset_vector_store

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
CORS(app)

# Routes
@app.route('/')
def index():
    """Render the dashboard page."""
    return render_template('index.html')

@app.route('/sms', methods=['POST'])
def sms_webhook():
    """Handle incoming SMS messages from Twilio."""
    try:
        # Extract message body from Twilio request
        incoming_msg = request.form.get('Body', '')
        logger.debug(f"Received message: {incoming_msg}")
        
        # Get sender's phone number
        from_number = request.form.get('From', '')
        logger.debug(f"From number: {from_number}")
        
        # Process the message and get a response
        response_text = process_message(incoming_msg)
        
        # Return response in TwiML format
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Message>{response_text}</Message>
        </Response>
        """
        
        return twiml_response, 200, {'Content-Type': 'text/xml'}
    except Exception as e:
        logger.error(f"Error processing SMS webhook: {str(e)}")
        # Still return a valid TwiML response in case of error
        error_response = """<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Message>Sorry, I encountered an error processing your request. Please try again later.</Message>
        </Response>
        """
        return error_response, 200, {'Content-Type': 'text/xml'}

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """API endpoint for web interface chat."""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
            
        response = process_message(user_message)
        return jsonify({"reply": response})
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_endpoint():
    """Reset the FAISS vector store."""
    try:
        result = reset_vector_store()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        return jsonify({"error": "Failed to reset vector store", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
