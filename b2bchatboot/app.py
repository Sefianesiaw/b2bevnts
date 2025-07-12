from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import time
import os
from datetime import datetime
import json
import threading
from functools import wraps

# Import du chatbot amélioré
try:
    from b2bchat import chatbot, invoke_chain, initialize_chatbot
except ImportError as e:
    logging.error(f"Erreur import chatbot: {e}")
    chatbot = None
    def invoke_chain(question):
        return "❌ Système de chat non disponible - Erreur de configuration"
    def initialize_chatbot():
        return False

# Configuration de l'application Flask
app = Flask(__name__)
CORS(app, origins=["*"])  # En production, spécifier les domaines autorisés

# Configuration des logs avec format amélioré
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting amélioré
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per hour", "20 per minute"],
    storage_uri="memory://"  # Utiliser la mémoire pour le stockage des limites
)

# Statistiques simples en mémoire
app_stats = {
    "requests_total": 0,
    "requests_successful": 0,
    "requests_failed": 0,
    "start_time": datetime.now(),
    "last_request": None
}

def update_stats(success=True):
    """Met à jour les statistiques de l'application"""
    app_stats["requests_total"] += 1
    app_stats["last_request"] = datetime.now()
    if success:
        app_stats["requests_successful"] += 1
    else:
        app_stats["requests_failed"] += 1

# Décorateur pour logging et stats
def log_and_stats(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        success = True
        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            logger.error(f"Erreur dans {f.__name__}: {str(e)}")
            raise
        finally:
            processing_time = time.time() - start_time
            update_stats(success)
            if request.endpoint != 'health_check':  # Éviter spam des health checks
                logger.info(f"{request.method} {request.path} - {processing_time:.2f}s - {'✅' if success else '❌'}")
    return decorated_function

@app.before_request
def before_request():
    """Middleware de pré-traitement des requêtes"""
    request.start_time = time.time()
    
    # Log uniquement les requêtes importantes
    if request.endpoint not in ['health_check']:
        logger.info(f"🔄 {request.method} {request.path} - {request.remote_addr}")

@app.after_request
def after_request(response):
    """Middleware de post-traitement avec headers de sécurité"""
    # Headers de sécurité
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Ajouter le temps de traitement dans les headers
    if hasattr(request, 'start_time'):
        processing_time = time.time() - request.start_time
        response.headers['X-Processing-Time'] = f"{processing_time:.3f}"
    
    return response

# Routes API améliorées

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état de santé de l'API avec détails étendus"""
    try:
        uptime = datetime.now() - app_stats["start_time"]
        
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.1",
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_human": str(uptime).split('.')[0],
            "chatbot_available": chatbot is not None,
            "stats": {
                "total_requests": app_stats["requests_total"],
                "successful_requests": app_stats["requests_successful"],
                "failed_requests": app_stats["requests_failed"],
                "success_rate": round(
                    (app_stats["requests_successful"] / max(app_stats["requests_total"], 1)) * 100, 2
                )
            }
        }
        
        # Test détaillé du chatbot si disponible
        if chatbot:
            try:
                chatbot_health = chatbot.health_check()
                status["chatbot_status"] = chatbot_health
                
                if chatbot_health.get("status") != "healthy":
                    status["status"] = "degraded"
                    
            except Exception as e:
                status["chatbot_status"] = {"status": "error", "error": str(e)}
                status["status"] = "degraded"
        else:
            status["status"] = "degraded"
            status["chatbot_status"] = {"status": "unavailable"}
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/ask', methods=['POST'])
@limiter.limit("10 per minute")
@log_and_stats
def ask_question():
    """Endpoint principal pour poser des questions au chatbot hybride"""
    try:
        # Validation du Content-Type
        if not request.is_json:
            return jsonify({
                "error": "Content-Type doit être application/json",
                "success": False,
                "code": "INVALID_CONTENT_TYPE"
            }), 400
        
        data = request.get_json()
        
        # Validation des données d'entrée
        if not data:
            return jsonify({
                "error": "Corps de requête JSON requis",
                "success": False,
                "code": "MISSING_BODY"
            }), 400
        
        question = data.get('question', '').strip()
        
        # Validation de la question
        if not question:
            return jsonify({
                "error": "Question vide ou manquante",
                "success": False,
                "code": "EMPTY_QUESTION"
            }), 400
        
        if len(question) > 2000:  # Augmenté pour permettre des questions plus longues
            return jsonify({
                "error": "Question trop longue (maximum 2000 caractères)",
                "success": False,
                "code": "QUESTION_TOO_LONG"
            }), 400
        
        # Vérification de la disponibilité du chatbot
        if chatbot is None:
            # Tentative de réinitialisation
            logger.info("🔄 Tentative de réinitialisation du chatbot...")
            if not initialize_chatbot():
                return jsonify({
                    "error": "Service de chat temporairement indisponible",
                    "success": False,
                    "code": "SERVICE_UNAVAILABLE",
                    "retry_after": 60
                }), 503
        
        logger.info(f"🤔 Traitement de la question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        # Traitement de la question avec timeout
        try:
            response = invoke_chain(question)
            processing_time = time.time() - request.start_time
            
            logger.info(f"✅ Question traitée en {processing_time:.2f}s")
            
            return jsonify({
                "response": response,
                "success": True,
                "processing_time": round(processing_time, 3),
                "timestamp": datetime.now().isoformat(),
                "question_length": len(question),
                "response_length": len(response) if response else 0
            }), 200
            
        except Exception as processing_error:
            logger.error(f"❌ Erreur de traitement: {str(processing_error)}", exc_info=True)
            return jsonify({
                "error": "Erreur lors du traitement de votre question",
                "success": False,
                "code": "PROCESSING_ERROR",
                "details": str(processing_error) if app.debug else "Erreur interne"
            }), 500
        
    except Exception as e:
        processing_time = time.time() - request.start_time if hasattr(request, 'start_time') else 0
        logger.error(f"❌ Erreur requête après {processing_time:.2f}s: {str(e)}", exc_info=True)
        
        return jsonify({
            "error": "Erreur technique inattendue",
            "success": False,
            "code": "UNEXPECTED_ERROR",
            "processing_time": round(processing_time, 3),
            "details": str(e) if app.debug else "Erreur interne"
        }), 500

@app.route('/batch', methods=['POST'])
@limiter.limit("2 per minute")
@log_and_stats
def batch_questions():
    """Endpoint pour traiter plusieurs questions en lot (amélioré)"""
    try:
        data = request.get_json()
        questions = data.get('questions', [])
        
        if not questions or not isinstance(questions, list):
            return jsonify({
                "error": "Liste de questions requise",
                "success": False,
                "code": "INVALID_QUESTIONS_LIST"
            }), 400
        
        if len(questions) > 3:  # Réduit pour éviter les timeouts
            return jsonify({
                "error": "Maximum 3 questions par lot",
                "success": False,
                "code": "TOO_MANY_QUESTIONS"
            }), 400
        
        results = []
        total_processing_time = 0
        
        for i, question in enumerate(questions):
            start_time = time.time()
            try:
                if isinstance(question, str) and question.strip():
                    response = invoke_chain(question.strip())
                    processing_time = time.time() - start_time
                    total_processing_time += processing_time
                    
                    results.append({
                        "index": i,
                        "question": question,
                        "response": response,
                        "success": True,
                        "processing_time": round(processing_time, 3)
                    })
                else:
                    results.append({
                        "index": i,
                        "question": question,
                        "error": "Question invalide ou vide",
                        "success": False,
                        "processing_time": 0
                    })
            except Exception as e:
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                results.append({
                    "index": i,
                    "question": question,
                    "error": str(e),
                    "success": False,
                    "processing_time": round(processing_time, 3)
                })
        
        successful_count = sum(1 for r in results if r.get("success"))
        
        return jsonify({
            "results": results,
            "summary": {
                "total": len(questions),
                "successful": successful_count,
                "failed": len(questions) - successful_count,
                "success_rate": round((successful_count / len(questions)) * 100, 2),
                "total_processing_time": round(total_processing_time, 3)
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement batch: {str(e)}")
        return jsonify({
            "error": "Erreur lors du traitement en lot",
            "success": False,
            "code": "BATCH_ERROR"
        }), 500

@app.route('/stats', methods=['GET'])
@log_and_stats
def get_stats():
    """Statistiques détaillées de l'API"""
    try:
        uptime = datetime.now() - app_stats["start_time"]
        
        stats = {
            "system": {
                "uptime_seconds": int(uptime.total_seconds()),
                "uptime_human": str(uptime).split('.')[0],
                "version": "2.1",
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            },
            "requests": {
                "total": app_stats["requests_total"],
                "successful": app_stats["requests_successful"],
                "failed": app_stats["requests_failed"],
                "success_rate": round(
                    (app_stats["requests_successful"] / max(app_stats["requests_total"], 1)) * 100, 2
                ),
                "last_request": app_stats["last_request"].isoformat() if app_stats["last_request"] else None
            },
            "chatbot": {
                "available": chatbot is not None,
                "status": "unknown"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Ajouter les statistiques détaillées du chatbot si disponible
        if chatbot:
            try:
                chatbot_health = chatbot.health_check()
                stats["chatbot"].update(chatbot_health)
            except Exception as e:
                stats["chatbot"]["status"] = "error"
                stats["chatbot"]["error"] = str(e)
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"❌ Erreur stats: {str(e)}")
        return jsonify({
            "error": "Erreur lors de la récupération des statistiques",
            "code": "STATS_ERROR"
        }), 500

@app.route('/debug', methods=['POST'])
@limiter.limit("5 per minute")
@log_and_stats
def debug_search():
    """Endpoint de debug pour tester la recherche dans le PDF"""
    if not app.debug and os.environ.get('FLASK_ENV') != 'development':
        return jsonify({
            "error": "Endpoint de debug disponible uniquement en mode développement",
            "success": False
        }), 403
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "error": "Question requise pour le debug",
                "success": False
            }), 400
        
        if chatbot is None:
            return jsonify({
                "error": "Chatbot non disponible",
                "success": False
            }), 503
        
        # Effectuer la recherche debug
        debug_results = []
        try:
            if hasattr(chatbot, 'debug_search'):
                # Capturer les logs pour les retourner
                import io
                import sys
                
                log_capture = io.StringIO()
                handler = logging.StreamHandler(log_capture)
                logger.addHandler(handler)
                
                chatbot.debug_search(question, k=5)
                
                logger.removeHandler(handler)
                debug_output = log_capture.getvalue()
                
                # Recherche directe pour les résultats
                docs = chatbot.vector_db.similarity_search(question, k=5) if chatbot.vector_db else []
                
                for i, doc in enumerate(docs):
                    debug_results.append({
                        "rank": i + 1,
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata,
                        "content_length": len(doc.page_content)
                    })
                
                return jsonify({
                    "question": question,
                    "documents_found": len(debug_results),
                    "documents": debug_results,
                    "debug_logs": debug_output,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }), 200
            else:
                return jsonify({
                    "error": "Fonction de debug non disponible",
                    "success": False
                }), 501
                
        except Exception as e:
            logger.error(f"Erreur debug search: {str(e)}")
            return jsonify({
                "error": "Erreur lors du debug",
                "details": str(e),
                "success": False
            }), 500
        
    except Exception as e:
        logger.error(f"Erreur endpoint debug: {str(e)}")
        return jsonify({
            "error": "Erreur technique du debug",
            "success": False
        }), 500

@app.route('/reinitialize', methods=['POST'])
@limiter.limit("1 per 5 minutes")
@log_and_stats
def reinitialize_chatbot():
    """Endpoint pour réinitialiser le chatbot manuellement"""
    try:
        logger.info("🔄 Réinitialisation manuelle du chatbot demandée...")
        
        global chatbot
        chatbot = None
        
        success = initialize_chatbot()
        
        if success:
            return jsonify({
                "message": "Chatbot réinitialisé avec succès",
                "success": True,
                "chatbot_available": chatbot is not None,
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                "message": "Échec de la réinitialisation du chatbot",
                "success": False,
                "chatbot_available": False,
                "timestamp": datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Erreur réinitialisation: {str(e)}")
        return jsonify({
            "error": "Erreur lors de la réinitialisation",
            "success": False,
            "details": str(e)
        }), 500

# Gestionnaires d'erreur globaux améliorés

@app.errorhandler(429)
def ratelimit_handler(e):
    """Gestionnaire pour les erreurs de rate limiting"""
    logger.warning(f"Rate limit atteint: {request.remote_addr} - {request.path}")
    return jsonify({
        "error": "Trop de requêtes. Veuillez patienter avant de réessayer.",
        "success": False,
        "code": "RATE_LIMIT_EXCEEDED",
        "retry_after": getattr(e, 'retry_after', 60),
        "description": "Vous avez dépassé la limite de requêtes autorisées."
    }), 429

@app.errorhandler(404)
def not_found(e):
    """Gestionnaire pour les routes non trouvées"""
    return jsonify({
        "error": "Endpoint non trouvé",
        "success": False,
        "code": "ENDPOINT_NOT_FOUND",
        "requested_path": request.path,
        "available_endpoints": {
            "POST /ask": "Poser une question au chatbot",
            "POST /batch": "Traiter plusieurs questions",
            "GET /health": "Vérifier l'état de santé",
            "GET /stats": "Obtenir les statistiques",
            "POST /debug": "Debug de recherche (dev seulement)",
            "POST /reinitialize": "Réinitialiser le chatbot"
        }
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Gestionnaire pour les méthodes non autorisées"""
    return jsonify({
        "error": f"Méthode {request.method} non autorisée pour {request.path}",
        "success": False,
        "code": "METHOD_NOT_ALLOWED",
        "allowed_methods": list(e.valid_methods) if hasattr(e, 'valid_methods') else []
    }), 405

@app.errorhandler(500)
def internal_error(e):
    """Gestionnaire pour les erreurs internes"""
    logger.error(f"Erreur interne: {str(e)}", exc_info=True)
    return jsonify({
        "error": "Erreur interne du serveur",
        "success": False,
        "code": "INTERNAL_SERVER_ERROR",
        "timestamp": datetime.now().isoformat()
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Gestionnaire d'exception global"""
    logger.error(f"Exception non gérée: {str(e)}", exc_info=True)
    return jsonify({
        "error": "Une erreur inattendue s'est produite",
        "success": False,
        "code": "UNHANDLED_EXCEPTION",
        "timestamp": datetime.now().isoformat()
    }), 500

# Configuration et démarrage

def setup_logging():
    """Configuration avancée des logs"""
    # Supprimer les handlers par défaut
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Format personnalisé
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Handler pour fichier
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configuration du logger racine
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

if __name__ == '__main__':
    # Configuration des logs
    setup_logging()
    
    # Configuration basée sur l'environnement
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    host = os.environ.get('HOST', '127.0.0.1')
    
    # Messages de démarrage
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE DE L'API B2B CHAT HYBRIDE")
    logger.info("=" * 60)
    logger.info(f"🌐 Serveur: {host}:{port}")
    logger.info(f"🔧 Mode debug: {debug}")
    logger.info(f"🤖 Chatbot disponible: {chatbot is not None}")
    logger.info(f"📁 Répertoire de travail: {os.getcwd()}")
    logger.info(f"🐍 Version Python: {os.sys.version}")
    
    if chatbot:
        logger.info("✅ Système prêt - Chatbot hybride opérationnel")
    else:
        logger.warning("⚠️ Système démarré en mode dégradé - Chatbot non disponible")
    
    logger.info("=" * 60)
    
    # Démarrage du serveur
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=debug  # Rechargement auto en mode debug seulement
        )
    except Exception as e:
        logger.error(f"💥 Erreur fatale au démarrage: {str(e)}")
        raise