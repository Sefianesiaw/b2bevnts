from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import time
import os
from datetime import datetime

# Configuration simple sans rate limiting pour débuter
app = Flask(__name__)
CORS(app)

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import du chatbot avec gestion d'erreur
chatbot = None
try:
    from b2bchat import chatbot as imported_chatbot
    chatbot = imported_chatbot
    logger.info("✅ Chatbot amélioré chargé")
except Exception as e:
    logger.warning(f"⚠️ Chatbot amélioré non disponible: {str(e)}")
    try:
        # Fallback vers l'ancien système
        from b2bchat import chain
        def invoke_chain(question):
            return chain.invoke(question)
        logger.info("✅ Chatbot original chargé")
    except Exception as e2:
        logger.error(f"❌ Aucun chatbot disponible: {str(e2)}")
        def invoke_chain(question):
            return "❌ Service de chat temporairement indisponible"

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification simple de l'état"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chatbot_available": chatbot is not None
    }), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint principal simplifié"""
    start_time = time.time()
    
    try:
        # Validation basique
        if not request.is_json:
            return jsonify({
                "error": "Content-Type doit être application/json",
                "success": False
            }), 400
        
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "error": "Question vide",
                "success": False
            }), 400
        
        logger.info(f"Question reçue: {question[:50]}...")
        
        # Traitement avec le chatbot disponible
        if chatbot:
            response = chatbot.ask(question)
        else:
            response = invoke_chain(question)
        
        processing_time = round(time.time() - start_time, 2)
        
        return jsonify({
            "response": response,
            "success": True,
            "processing_time": processing_time
        }), 200
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        logger.error(f"Erreur: {str(e)}")
        
        return jsonify({
            "error": "Erreur lors du traitement",
            "success": False,
            "processing_time": processing_time
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint non trouvé",
        "available_endpoints": ["/ask", "/health"]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Erreur interne du serveur"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"🚀 Démarrage de l'API sur le port {port}")
    logger.info(f"🤖 Chatbot disponible: {chatbot is not None}")
    
    app.run(
        host='127.0.0.1',
        port=port,
        debug=debug
    )