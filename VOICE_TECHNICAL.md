# Guide Technique - Système de Commandes Vocales IRIS

## Installation et Configuration

### Fichiers Modifiés/Créés

1. **`interface/js/voice-commands.js`** - Module principal de reconnaissance vocale
   - Classe `VoiceController` pour la gestion de la reconnaissance vocale
   - Fonctions `speak()` pour la synthèse vocale
   - Traitement des commandes vocales

2. **`interface/html/index.html`** - Interface principale
   - Ajout de l'indicateur vocal (#voiceIndicator)
   - Intégration du module voice-commands.js
   - Modification de `startPipeline()` et `stopPipeline()` pour appeler les fonctions vocales

3. **`interface/html/about.html`** - Page À Propos
   - Ajout de contenu HTML pour la lecture vocale
   - Intégration du module voice-commands.js

4. **`interface/html/contact.html`** - Page Contact
   - Ajout de contenu HTML pour la lecture vocale
   - Intégration du module voice-commands.js

5. **`interface/html/closedones.html`** - Page Entourage
   - Intégration du module voice-commands.js

6. **`interface/html/settings.html`** - Page Paramètres
   - Intégration du module voice-commands.js

7. **`interface/html/visualdeficiency.html`** - Page Profil Visuel
   - Intégration du module voice-commands.js

## Architecture du Système Vocal

### Reconnaissance Vocale (Speech Recognition)

```
[Utilisateur parle] 
    ↓
[Web Speech API - SpeechRecognition]
    ↓
[Transcription en temps réel]
    ↓
[VoiceController.processCommand()]
    ↓
[Matching avec les commandes]
    ↓
[Exécution de l'action]
```

### Synthèse Vocale (Text-to-Speech)

```
[Texte à lire]
    ↓
[speak() function]
    ↓
[Web Speech API - SpeechSynthesis]
    ↓
[Audio en français]
    ↓
[Utilisateur entend]
```

### Flux de Commande Vocal

1. **Détection** - L'utilisateur parle
2. **Reconnaissance** - Speech API transcrit en texte
3. **Normalisation** - Conversion en minuscules et trim
4. **Matching** - Comparaison avec les mots-clés connus
5. **Exécution** - Appel de la fonction correspondante
6. **Retour** - Confirmation audio et mise à jour de l'état

## Implémentation des Commandes

### Structure de Traitement

```javascript
// Chaque commande utilise matchesKeywords()
if (this.matchesKeywords(command, ['mot1', 'mot2', 'phrase'])) {
    this.executeAction();
}
```

### Exemple: Ajouter une Commande

```javascript
// Dans VoiceController.processCommand()
else if (this.matchesKeywords(command, ['ma commande', 'autres variantes'])) {
    this.executeMyCommand();
    commandExecuted = true;
}

// Ajouter la fonction d'exécution
executeMyCommand() {
    speak('Confirmation audio de l\'action');
    // ... effectuer l'action
}
```

## Configuration de la Langue

Le système est configuré pour le français par défaut :

```javascript
this.recognition.lang = 'fr-FR';  // Reconnaissance
utterance.lang = 'fr-FR';         // Synthèse
```

Pour ajouter le support de l'anglais :

```javascript
// Détecter la langue de l'utilisateur
const userLang = navigator.language || navigator.userLanguage;
this.recognition.lang = userLang.startsWith('fr') ? 'fr-FR' : 'en-US';
```

## Gestion des États

L'indicateur vocal a plusieurs états :

| État | Couleur | Signification |
|------|---------|--------------|
| ready | Cyan (#00BDC6) | Prêt à écouter |
| listening | Vert (#00FF00) | Enregistrement en cours |
| processing | Orange (#FFA500) | Traitement de la commande |
| error | Rouge (#FF6B6B) | Erreur détectée |

## Cycle de Vie

### Au démarrage de la page
1. `DOMContentLoaded` → `initVoiceControl()`
2. Création de l'instance `VoiceController`
3. Vérification de la compatibilité du navigateur
4. Mise à jour de l'indicateur

### Au démarrage du pipeline
1. Utilisateur clique "Start"
2. `startPipeline()` appelé
3. `startVoiceControl()` activée
4. Recognition commence à écouter
5. Message audio: "Commandes vocales activées..."

### Lors d'une commande vocale
1. `onresult` déclenché
2. Transcription reçue
3. `processCommand()` traite la commande
4. Action exécutée
5. Retour audio fourni
6. Après 2 secondes, recommence à écouter

### À l'arrêt du pipeline
1. Utilisateur clique "Stop"
2. `stopPipeline()` appelé
3. `stopVoiceControl()` désactive l'écoute
4. Indicateur revient à l'état initial

## Gestion des Erreurs

### Erreurs de Microphone
- **Pas d'accès au microphone** : Message d'erreur dans la console
- **Microphone désactivé** : Vérifier les permissions du navigateur
- **Pas d'audio** : Vérifier le volume du système

### Erreurs de Reconnaissance
- **"no-speech"** : Aucune voix détectée dans le délai imparti
- **"network"** : Problème de connectivité (moins fréquent)
- **"aborted"** : Reconnaissance stoppée intentionnellement

### Gestion du Code

```javascript
this.recognition.onerror = (event) => {
    console.error('Voice recognition error:', event.error);
    if (event.error === 'network') {
        this.updateVoiceIndicator('error', 'Erreur réseau');
    } else if (event.error === 'no-speech') {
        this.updateVoiceIndicator('ready', 'Pas de voix détectée');
    }
};
```

## Test et Débogage

### Vérifier la Compatibilité

```javascript
// Dans la console du navigateur
window.SpeechRecognition || window.webkitSpeechRecognition
// Devrait retourner une fonction, pas undefined
```

### Test des Commandes

1. **Via la console** :
```javascript
voiceController.processCommand("démarrer");
```

2. **Via la parole** :
- Parlez clairement près du microphone
- Attendez la transcription finale
- L'action devrait s'exécuter

### Logs de Débogage

```javascript
// Vérifier les logs dans la console F12
console.log('Voice command received:', command);
console.log('Reading page content, length:', fullText.length);
console.log('Voice recognition started');
```

## Performance

### Optimisations Appliquées

1. **Continuous Recognition** - `continuous: true` pour une écoute continue
2. **Interim Results** - `interimResults: true` pour le retour en temps réel
3. **Lazy Initialization** - VoiceController créé seulement si nécessaire
4. **Automatic Restart** - Redémarrage automatique après chaque commande
5. **Page Visibility API** - Arrêt de l'écoute quand la page est cachée

### Limitations Connues

- Pas de support pour les commandes très longues (> 5 secondes)
- La synthèse vocale peut être interrompue si plusieurs speak() sont appelés
- La reconnaissance vocale dépend de la qualité du microphone

## Support des Navigateurs

| Navigateur | Version | Support |
|-----------|---------|---------|
| Chrome | 27+ | ✓ Complet |
| Edge | 79+ | ✓ Complet |
| Firefox | 25+ | ✓ Avec limites |
| Safari | 14.1+ | ✓ Complet |
| Opera | 15+ | ✓ Complet |
| IE | Tous | ✗ Non supporté |

## API Utilisées

### Web Speech API - SpeechRecognition
- `start()` - Démarre la reconnaissance
- `stop()` - Arrête la reconnaissance
- `abort()` - Annule l'opération en cours
- Événements: `onstart`, `onresult`, `onerror`, `onend`

### Web Speech API - SpeechSynthesis
- `speak(utterance)` - Parle le texte
- `cancel()` - Annule la parole en cours
- Configuration: `lang`, `rate`, `pitch`, `volume`

## Sécurité et Confidentialité

1. **Traitement Local** - Tout est traité côté client
2. **Pas de Transmission de Données** - Les commandes ne sont jamais envoyées à des serveurs externes
3. **Permissions** - Le navigateur demande l'autorisation d'accès au microphone
4. **Conformité RGPD** - Aucune donnée personnelle n'est stockée

## Améliorations Futures

1. **Commandes Composées** - "Ajoute Marie et Jean à l'entourage"
2. **Reconnaissance de Noms** - "Dis-moi qui est devant moi"
3. **Commandes Contextuelles** - Différentes commandes selon la page
4. **Apprentissage** - Adaptation à l'accent de l'utilisateur
5. **Multi-langue** - Support automatique du français et anglais
6. **Commandes Avancées** - Paramètres de synthèse vocale

## Référence API

### VoiceController

**Constructor**
```javascript
const vc = new VoiceController();
```

**Méthodes Publiques**
```javascript
vc.start()                      // Démarre l'écoute
vc.stop()                       // Arrête l'écoute
vc.restart()                    // Redémarre l'écoute
vc.processCommand(text)         // Traite une commande
```

**Propriétés**
```javascript
vc.supported              // boolean - Support du navigateur
vc.isListening            // boolean - État d'écoute
vc.recognition            // SpeechRecognition - Objet API
```

### Fonctions Globales

```javascript
initVoiceControl()        // Initialise le contrôle vocal
startVoiceControl()       // Démarre l'écoute vocale
stopVoiceControl()        // Arrête l'écoute vocale
speak(text)              // Parle le texte fourni
```

---

**Version:** 1.0
**Dernière mise à jour:** Mai 2026
**Support:** support@iris-vision.org
