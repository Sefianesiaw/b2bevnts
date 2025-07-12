import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from functools import lru_cache
import hashlib

from PyPDF2 import PdfReader
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from dotenv import load_dotenv
import re

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class B2BChatbot:
    def __init__(self, pdf_path: str = "data/b2b.pdf", persist_directory: str = "chroma_db2"):
        """Initialise le chatbot B2B avec gestion hybride PDF + connaissances générales"""
        load_dotenv()
        
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        
        # Configuration optimisée pour meilleure récupération
        self.chunk_size = 1000
        self.chunk_overlap = 250
        self.top_k = 8
        
        # Initialisation des composants
        self.embeddings = None
        self.vector_db = None
        self.llm = None
        self.hybrid_chain = None
        self.pdf_content = ""
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialise tous les composants avec gestion d'erreurs améliorée"""
        try:
            self._setup_llm()
            self._setup_embeddings()
            self._setup_vector_db()
            self._setup_hybrid_chain()
            logger.info("✅ Chatbot B2B initialisé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
            raise
    
    def _setup_llm(self):
        """Configure le modèle de langage avec paramètres optimisés"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY non trouvée dans l'environnement")
            
            self.llm = ChatGroq(
                model_name="llama3-8b-8192",
                api_key=api_key,
                temperature=0.1,  # Plus déterministe pour les réponses factuelles
                max_tokens=1500,  # Augmenté pour des réponses plus complètes
                timeout=45,
                streaming=False
            )
            logger.info("✅ LLM configuré avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur configuration LLM: {str(e)}")
            raise
    
    def _setup_embeddings(self):
        """Configure les embeddings avec gestion d'erreur"""
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )
            # Test des embeddings
            test_embedding = self.embeddings.embed_query("test")
            logger.info("✅ Embeddings configurés et testés")
        except Exception as e:
            logger.error(f"❌ Erreur configuration embeddings: {str(e)}")
            logger.info("💡 Assurez-vous qu'Ollama est installé et que le modèle nomic-embed-text est disponible")
            raise
    
    def _setup_vector_db(self):
        """Configure la base vectorielle avec traitement PDF amélioré"""
        try:
            # Vérifier si la DB existe et si le PDF a changé
            pdf_hash = self._get_pdf_hash()
            hash_file = os.path.join(self.persist_directory, "pdf_hash.txt")
            
            should_recreate = True
            if os.path.exists(self.persist_directory) and os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    stored_hash = f.read().strip()
                if stored_hash == pdf_hash:
                    should_recreate = False
                    logger.info("📁 Base vectorielle existante et à jour")
            
            if should_recreate:
                logger.info("🔄 Création/mise à jour de la base vectorielle...")
                self._create_vector_db()
                # Sauvegarder le hash
                os.makedirs(self.persist_directory, exist_ok=True)
                with open(hash_file, 'w') as f:
                    f.write(pdf_hash)
            else:
                self.vector_db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            
            logger.info("✅ Base vectorielle prête")
        except Exception as e:
            logger.error(f"❌ Erreur setup vector DB: {str(e)}")
            raise
    
    def _get_pdf_hash(self):
        """Calcule le hash du PDF pour détecter les changements"""
        try:
            with open(self.pdf_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return "no_pdf"
    
    def _create_vector_db(self):
        """Crée la base vectorielle avec extraction PDF améliorée"""
        try:
            if not os.path.exists(self.pdf_path):
                logger.warning(f"⚠️ PDF non trouvé: {self.pdf_path}")
                # Créer une base vide pour continuer à fonctionner
                self.vector_db = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                return
            
            logger.info(f"📖 Lecture et traitement du PDF: {self.pdf_path}")
            reader = PdfReader(self.pdf_path)
            
            # Extraction du texte avec nettoyage
            text_parts = []
            metadata_parts = []
            
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Nettoyage du texte
                        cleaned_text = self._clean_text(page_text)
                        text_parts.append(cleaned_text)
                        metadata_parts.append({"page": i + 1, "source": "pdf"})
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lecture page {i+1}: {str(e)}")
                    continue
            
            if not text_parts:
                raise ValueError("Aucun texte extrait du PDF")
            
            # Conserver le contenu complet pour référence
            self.pdf_content = "\n\n--- Page {} ---\n".join(
                [f"{i+1} ---\n{text}" for i, text in enumerate(text_parts)]
            )
            
            # Découpage intelligent du texte
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", ".", "!", "?", ";", ":", " "],
                keep_separator=True
            )
            
            # Créer les documents avec métadonnées
            documents = []
            for i, (text, metadata) in enumerate(zip(text_parts, metadata_parts)):
                page_doc = Document(page_content=text, metadata=metadata)
                documents.append(page_doc)
            
            # Découper les documents
            chunks = text_splitter.split_documents(documents)
            
            logger.info(f"📄 {len(chunks)} chunks créés à partir de {len(text_parts)} pages")
            
            # Création de la base vectorielle
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="b2b_knowledge",
                persist_directory=self.persist_directory
            )
            self.vector_db.persist()
            logger.info("💾 Base vectorielle créée et sauvegardée")
            
        except Exception as e:
            logger.error(f"❌ Erreur création vector DB: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte extrait du PDF"""
        # Supprimer les caractères de contrôle et normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\'\"]+', ' ', text)
        return text.strip()
    
    def _setup_hybrid_chain(self):
        """Configure une chaîne hybride qui combine PDF et connaissances générales"""
        try:
            # Template hybride amélioré
            template = """Tu es un expert consultant B2B spécialisé dans la taxonomie et les stratégies business.

CONTEXTE DOCUMENTAIRE:
{pdf_context}

QUESTION: {question}

INSTRUCTIONS:
1. D'ABORD, vérifie si l'information est disponible dans le contexte documentaire ci-dessus
2. Si l'information est dans le document, utilise-la comme base principale de ta réponse
3. Si l'information n'est pas complète dans le document, complète avec tes connaissances expertes en B2B
4. Si l'information n'est pas du tout dans le document, réponds basé sur ton expertise B2B
5. Sois toujours précis, professionnel et utile
6. Réponds en français de manière structurée et professionnelle

RÉPONSE EXPERTE:"""

            prompt = ChatPromptTemplate.from_template(template)
            
            # Créer le retriever si la base vectorielle existe
            if self.vector_db:
                retriever = self.vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.top_k}
                )
                
                self.hybrid_chain = (
                    {"pdf_context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
            else:
                # Chain sans PDF (utilise seulement les connaissances générales)
                fallback_template = """Tu es un expert consultant B2B. Réponds à cette question avec ton expertise:

QUESTION: {question}

RÉPONSE EXPERTE:"""
                
                fallback_prompt = ChatPromptTemplate.from_template(fallback_template)
                self.hybrid_chain = (
                    {"question": RunnablePassthrough()}
                    | fallback_prompt
                    | self.llm
                    | StrOutputParser()
                )
            
            logger.info("✅ Chaîne hybride configurée")
        except Exception as e:
            logger.error(f"❌ Erreur setup chaîne hybride: {str(e)}")
            raise
    
    def ask(self, question: str) -> str:
        """Pose une question au chatbot hybride"""
        try:
            if not question or not question.strip():
                return "❌ Question vide. Veuillez poser une question."
            
            question = question.strip()
            logger.info(f"🤔 Question reçue: {question[:100]}...")
            
            # Utiliser la chaîne hybride
            response = self.hybrid_chain.invoke(question)
            
            if not response or response.strip() == "":
                return "❌ Désolé, je n'ai pas pu générer une réponse pertinente. Pouvez-vous reformuler votre question ?"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la question: {str(e)}")
            return "❌ Désolé, une erreur technique s'est produite. Veuillez réessayer dans quelques instants."
    
    def health_check(self) -> Dict:
        """Vérification de l'état de santé du chatbot"""
        try:
            status = {
                "llm_available": self.llm is not None,
                "embeddings_available": self.embeddings is not None,
                "vector_db_available": self.vector_db is not None,
                "pdf_loaded": os.path.exists(self.pdf_path) if self.pdf_path else False,
                "status": "healthy"
            }
            
            # Test rapide du LLM
            if self.llm:
                try:
                    test_response = self.llm.invoke("Test")
                    status["llm_responsive"] = True
                except:
                    status["llm_responsive"] = False
                    status["status"] = "degraded"
            
            return status
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def debug_search(self, question: str, k: int = 5):
        """Méthode de debug pour voir les documents récupérés"""
        if not self.vector_db:
            logger.info("❌ Pas de base vectorielle disponible")
            return
        
        try:
            docs = self.vector_db.similarity_search(question, k=k)
            logger.info(f"\n🔍 DEBUG - Question: {question}")
            logger.info(f"📊 {len(docs)} documents trouvés:")
            
            for i, doc in enumerate(docs):
                logger.info(f"\n--- Document {i+1} ---")
                logger.info(f"Page: {doc.metadata.get('page', 'N/A')}")
                logger.info(f"Contenu: {doc.page_content[:200]}...")
        except Exception as e:
            logger.error(f"❌ Erreur debug: {str(e)}")

# Instance globale avec gestion d'erreur robuste
chatbot = None

def initialize_chatbot():
    """Initialise le chatbot de manière sécurisée"""
    global chatbot
    try:
        chatbot = B2BChatbot()
        logger.info("🚀 Chatbot B2B initialisé avec succès!")
        return True
    except Exception as e:
        logger.error(f"💥 Échec d'initialisation du chatbot: {str(e)}")
        logger.info("💡 Le système fonctionnera en mode dégradé avec les connaissances générales uniquement")
        chatbot = None
        return False

# Initialisation au chargement du module
initialize_chatbot()

def invoke_chain(question: str) -> str:
    """Fonction d'interface pour l'API"""
    if chatbot is None:
        # Essayer de réinitialiser
        if not initialize_chatbot():
            return "❌ Système temporairement indisponible. L'équipe technique a été notifiée."
    
    return chatbot.ask(question)