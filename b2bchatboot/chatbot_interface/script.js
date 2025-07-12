document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = document.getElementById('typingIndicator');

    // Configuration de l'API
    const API_URL = 'http://localhost:5000/ask'; // Remplacez par votre endpoint
    
    // Envoyer un message
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Ajouter le message de l'utilisateur
        addMessage(message, 'user');
        userInput.value = '';
        
        // Afficher l'indicateur de saisie
        showTypingIndicator();
        
        // Envoyer au backend
        fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: message })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            addMessage(data.response, 'bot');
        })
        .catch(error => {
            hideTypingIndicator();
            addMessage("Désolé, une erreur s'est produite. Veuillez réessayer.", 'bot');
            console.error('Error:', error);
        });
    }

    // Ajouter un message au chat
    function addMessage(content, sender) {
        const now = new Date();
        const timeString = now.getHours() + ':' + (now.getMinutes() < 10 ? '0' : '') + now.getMinutes();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
            <div class="message-time">${timeString}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Afficher l'indicateur de saisie
    function showTypingIndicator() {
        typingIndicator.style.display = 'flex';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Cacher l'indicateur de saisie
    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }

    // Événements
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
document.addEventListener("DOMContentLoaded", function () {
    const chatToggle = document.getElementById("chatToggle");
    const chatContainer = document.getElementById("chatContainer");
    const chatIcon = chatToggle.querySelector(".chat-icon");
    const closeIcon = chatToggle.querySelector(".close-icon");
    const notificationBadge = document.getElementById("notificationBadge");

    chatToggle.addEventListener("click", function () {
        const isOpen = chatContainer.style.display === "flex";
        
        if (isOpen) {
            chatContainer.style.display = "none";
            chatIcon.classList.remove("hidden");
            closeIcon.classList.add("hidden");
        } else {
            chatContainer.style.display = "flex";
            chatIcon.classList.add("hidden");
            closeIcon.classList.remove("hidden");
            notificationBadge.style.display = "none";
        }
    });
});
