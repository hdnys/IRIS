/**
 * Voice Control Module for IRIS
 * Enables voice commands to control the interface
 * Supports French and English commands
 */

class VoiceController {
    constructor() {
        // Check browser support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn('Speech Recognition not supported in this browser');
            this.supported = false;
            return;
        }
        this.supported = true;
        this.recognition = new SpeechRecognition();
        this.isListening = false;
        this.lastCommand = '';
        this.commandTimeout = null;
        this.transcript = '';
        
        // Configuration
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'fr-FR';
        
        this.setupRecognitionHandlers();
    }

    setupRecognitionHandlers() {
        if (!this.supported) return;

        this.recognition.onstart = () => {
            this.isListening = true;
            this.transcript = '';
            this.updateVoiceIndicator('listening', '');
            console.log('Voice recognition started');
        };

        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i].transcript;
                
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interimTranscript += transcript;
                }
            }

            // Display interim results
            if (interimTranscript) {
                this.updateVoiceIndicator('listening', interimTranscript);
            }

            // Process final results
            if (finalTranscript) {
                this.transcript = finalTranscript;
                this.processCommand(finalTranscript);
            }
        };

        this.recognition.onerror = (event) => {
            console.error('Voice recognition error:', event.error);
            if (event.error === 'network') {
                this.updateVoiceIndicator('error', 'Erreur réseau');
            } else if (event.error === 'no-speech') {
                this.updateVoiceIndicator('ready', 'Pas de voix détectée');
            } else if (event.error !== 'aborted') {
                this.updateVoiceIndicator('error', event.error);
            }
        };

        this.recognition.onend = () => {
            this.isListening = false;
            // Don't reset to ready immediately - let the command handler manage this
            console.log('Voice recognition ended');
        };
    }

    start() {
        if (!this.supported) {
            console.warn('Voice recognition not supported');
            return false;
        }
        if (!this.isListening) {
            this.recognition.start();
        }
        return true;
    }

    stop() {
        if (this.isListening) {
            this.recognition.stop();
        }
    }

    restart() {
        this.stop();
        setTimeout(() => this.start(), 100);
    }

    processCommand(transcript) {
        // Normalize the transcript
        const command = transcript.toLowerCase().trim();
        this.lastCommand = command;
        
        console.log('Voice command received:', command);
        this.updateVoiceIndicator('processing', command);

        // Clear previous timeout
        if (this.commandTimeout) clearTimeout(this.commandTimeout);

        let commandExecuted = false;

        // Command matching (French and English)
        // Start/Stop commands
        if (this.matchesKeywords(command, ['démarrer', 'start', 'commencer', 'lancer'])) {
            this.executeStart();
            commandExecuted = true;
        }
        else if (this.matchesKeywords(command, ['arrêter', 'stop', 'stopper', 'éteindre'])) {
            this.executeStop();
            commandExecuted = true;
        }
        // Add to entourage
        else if (this.matchesKeywords(command, ['ajouter', 'add', 'ajoute', 'nouveau', 'ajoute quelqu\'un'])) {
            this.executeAddEntourage();
            commandExecuted = true;
        }
        // Go to settings
        else if (this.matchesKeywords(command, ['paramètres', 'settings', 'réglages', 'configuration', 'parametre'])) {
            this.executeGoTo('settings');
            commandExecuted = true;
        }
        // Go to visual deficiency settings
        else if (this.matchesKeywords(command, ['visual', 'vision', 'défaut', 'deficiency', 'défaut de vision', 'profil', 'visuel'])) {
            this.executeGoTo('visualdeficiency');
            commandExecuted = true;
        }
        // Go to about
        else if (this.matchesKeywords(command, ['about', 'à propos', 'apropos', 'information', 'quoi'])) {
            this.executeReadPage('about');
            commandExecuted = true;
        }
        // Go to contact
        else if (this.matchesKeywords(command, ['contact', 'contacter', 'nous contacter', 'nous joindre', 'joindre'])) {
            this.executeReadPage('contact');
            commandExecuted = true;
        }
        // Go to entourage
        else if (this.matchesKeywords(command, ['entourage', 'amis', 'friends', 'mes proches', 'proches', 'closedones'])) {
            this.executeReadPage('entourage');
            commandExecuted = true;
        }
        // Go home
        else if (this.matchesKeywords(command, ['accueil', 'home', 'maison', 'retour', 'iris'])) {
            this.executeGoTo('home');
            commandExecuted = true;
        }
        // Help
        else if (this.matchesKeywords(command, ['aide', 'help', 'quoi', 'commandes', 'comment'])) {
            this.showHelp();
            commandExecuted = true;
        }
        // Unknown command
        else {
            speak('Je n\'ai pas compris cette commande. Dites "aide" pour les commandes disponibles.');
            this.updateVoiceIndicator('ready', 'Commande non reconnue: ' + command);
            commandExecuted = true;
        }

        // Reset indicator after a moment
        if (commandExecuted) {
            this.commandTimeout = setTimeout(() => {
                this.updateVoiceIndicator('ready');
                // Automatically restart listening if pipeline is running
                if (document.getElementById('status')?.textContent === 'running') {
                    this.restart();
                }
            }, 2000);
        }
    }

    matchesKeywords(command, keywords) {
        return keywords.some(keyword => command.includes(keyword.toLowerCase()));
    }

    executeStart() {
        const btn = document.getElementById('startBtn');
        if (btn && !btn.disabled) {
            speak('Démarrage du pipeline.');
            btn.click();
        } else {
            speak('Le pipeline est déjà en cours d\'exécution.');
        }
    }

    executeStop() {
        const btn = document.getElementById('stopBtn');
        if (btn && !btn.disabled) {
            speak('Arrêt du pipeline.');
            btn.click();
        } else {
            speak('Le pipeline n\'est pas en cours d\'exécution.');
        }
    }

    executeAddEntourage() {
        const modal = document.getElementById('learnModal');
        if (modal && typeof openLearnModal !== 'undefined') {
            speak('Ajout à l\'entourage. Veuillez dire votre nom.');
            openLearnModal();
            
            // Start listening for the name after a brief delay
            setTimeout(() => {
                this.listeningForName = true;
                const input = document.getElementById('learnName');
                if (input) input.focus();
            }, 1000);
        } else {
            speak('La modal d\'ajout n\'est pas disponible sur cette page.');
        }
    }

    executeGoTo(page) {
        const routes = {
            'settings': '/interface/html/settings.html',
            'visualdeficiency': '/interface/html/visualdeficiency.html',
            'home': '/interface/html/index.html'
        };
        
        const url = routes[page];
        if (url) {
            const pageNames = {
                'settings': 'paramètres',
                'visualdeficiency': 'profil de vision',
                'home': 'accueil'
            };
            speak(`Redirection vers ${pageNames[page]}.`);
            setTimeout(() => window.location.href = url, 800);
        }
    }

    executeReadPage(page) {
        const pages = {
            'about': { url: '/interface/html/about.html', name: 'À Propos' },
            'contact': { url: '/interface/html/contact.html', name: 'Contact' },
            'entourage': { url: '/interface/html/closedones.html', name: 'Entourage' }
        };
        
        const pageInfo = pages[page];
        if (!pageInfo) return;

        speak(`Lecture de ${pageInfo.name}.`);
        
        // Fetch the page content and extract text
        fetch(pageInfo.url)
            .then(res => res.text())
            .then(html => {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                
                // Extract main content from container
                let content = doc.querySelector('.content-section');
                if (!content) content = doc.querySelector('.container');
                
                if (content) {
                    let text = content.textContent.trim();
                    if (text) {
                        this.readPageContent(pageInfo.name, text);
                    } else {
                        speak(`La page ${pageInfo.name} n'a pas de contenu disponible.`);
                    }
                } else {
                    speak(`La page ${pageInfo.name} n'a pas pu être lue.`);
                }
            })
            .catch(err => {
                console.error('Error reading page:', err);
                speak(`Erreur lors de la lecture de ${pageInfo.name}.`);
            });
    }

    readPageContent(pageName, text) {
        // Clean up the text
        text = text.replace(/\s+/g, ' ').trim();
        
        // Truncate if too long (more than 5000 chars)
        if (text.length > 5000) {
            text = text.substring(0, 5000) + '...';
        }
        
        const fullText = `${pageName}. ${text}`;
        
        console.log('Reading page content, length:', fullText.length);
        speak(fullText);
    }

    showHelp() {
        const helpText = `
            Commandes vocales disponibles.
            Dire démarrer pour lancer le pipeline.
            Dire arrêter pour arrêter le pipeline.
            Dire ajouter pour ajouter quelqu'un à l'entourage.
            Dire paramètres pour accéder aux réglages.
            Dire vision pour modifier le profil de défaut de vision.
            Dire à propos pour lire la page à propos.
            Dire contact pour lire la page contact.
            Dire entourage pour lire la liste de vos proches.
            Dire accueil pour retourner à la page d'accueil.
        `;
        speak(helpText);
    }

    updateVoiceIndicator(state, text = '') {
        let indicator = document.getElementById('voiceIndicator');
        if (!indicator) return;

        const states = {
            'ready': { color: '#00BDC6', icon: '🎤', label: 'Prêt' },
            'listening': { color: '#00FF00', icon: '🎤', label: 'Écoute' },
            'processing': { color: '#FFA500', icon: '🎤', label: 'Traitement' },
            'error': { color: '#FF6B6B', icon: '⚠️', label: 'Erreur' }
        };

        const stateInfo = states[state] || states['ready'];
        indicator.style.color = stateInfo.color;
        indicator.textContent = `${stateInfo.icon} ${stateInfo.label}`;
        
        // Show full text in title if it's longer
        if (text) {
            indicator.title = text.length > 50 ? text.substring(0, 47) + '...' : text;
        }
    }
}

// Global TTS function (extended from existing speak function)
let _lastSpoken = '';
function speak(text) {
    if (!window.speechSynthesis || !text) return;
    
    // Avoid repeating the exact same text immediately
    if (text === _lastSpoken) {
        console.log('Skipping duplicate TTS:', text.substring(0, 50));
        return;
    }
    
    _lastSpoken = text;
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    // Set language to French
    utterance.lang = 'fr-FR';
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    
    window.speechSynthesis.speak(utterance);
}

// Initialize voice controller
let voiceController = null;

function initVoiceControl() {
    if (voiceController) return; // Already initialized
    
    voiceController = new VoiceController();
    if (!voiceController.supported) {
        console.warn('Voice recognition not supported');
        const indicator = document.getElementById('voiceIndicator');
        if (indicator) {
            indicator.textContent = '⚠️ Voice not supported';
            indicator.style.color = '#999';
            indicator.title = 'Voice recognition is not supported in this browser';
        }
        return false;
    }
    return true;
}

// Start voice control when pipeline starts
function startVoiceControl() {
    if (!voiceController) {
        initVoiceControl();
    }
    
    if (voiceController && voiceController.supported) {
        voiceController.start();
        speak('Commandes vocales activées. Dites aide pour les commandes disponibles.');
    }
}

// Stop voice control when pipeline stops
function stopVoiceControl() {
    if (voiceController) {
        voiceController.stop();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initVoiceControl();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopVoiceControl();
    } else {
        // Resume if pipeline was running
        const status = document.getElementById('status');
        if (status && status.textContent === 'running') {
            startVoiceControl();
        }
    }
});

