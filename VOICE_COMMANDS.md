# Commandes Vocales IRIS

## Vue d'ensemble

L'interface IRIS dispose maintenant d'un système de reconnaissance vocale complet qui vous permet de contrôler l'application entièrement par la voix en français.

## Activation des Commandes Vocales

Les commandes vocales s'activent automatiquement lorsque vous :
1. Démarrez le pipeline avec le bouton **Start**
2. Dites "démarrer" ou "start"

Un indicateur vocal apparaîtra en bas à droite de l'écran pour vous informer de l'état d'écoute.

## Commandes Disponibles

### Contrôle du Pipeline

| Commande | Action |
|----------|--------|
| "démarrer", "start", "commencer", "lancer" | Démarre le pipeline |
| "arrêter", "stop", "stopper", "éteindre" | Arrête le pipeline |

### Navigation

| Commande | Action |
|----------|--------|
| "accueil", "home", "maison", "retour" | Retourne à la page d'accueil |
| "paramètres", "settings", "réglages", "configuration" | Accède aux paramètres |
| "vision", "visual", "défaut", "deficiency" | Modifie le profil de défaut de vision |
| "ajouter", "add", "nouveau" | Ajoute quelqu'un à l'entourage |

### Lecture de Pages

| Commande | Action |
|----------|--------|
| "à propos", "about", "information" | Lit la page À Propos |
| "contact", "nous contacter" | Lit la page Contact |
| "entourage", "amis", "friends", "mes proches" | Lit la page Entourage |

### Autres

| Commande | Action |
|----------|--------|
| "aide", "help", "commandes" | Affiche les commandes disponibles |

## Fonctionnement

### Indicateur Vocal

L'indicateur vocal en bas à droite vous montre l'état du système :

- 🎤 **Prêt à écouter** (Bleu cyan) - Le système attend une commande
- 🎤 **Écoute...** (Vert) - Le système enregistre votre voix
- 🎤 **Traitement...** (Orange) - Le système analyse votre commande
- ⚠️ **Voice not supported** (Gris) - Votre navigateur ne supporte pas la reconnaissance vocale

### Ajouter quelqu'un à l'Entourage par Voix

1. Dites "ajouter" pour ouvrir la modale d'ajout
2. Dites votre nom quand vous êtes invité
3. Le système commencera à apprendre votre visage
4. Regardez la caméra et suivez les instructions affichées

### Lire une Page par Voix

Quand vous dites "à propos", "contact" ou "entourage", le système :
1. Charge la page demandée
2. Extrait le contenu texte
3. Utilise la synthèse vocale pour vous lire le contenu

L'audio est lu en français.

## Support des Navigateurs

La reconnaissance vocale fonctionne dans les navigateurs suivants :
- Chrome / Chromium ✓
- Firefox ✓ (avec contrôles supplémentaires)
- Safari ✓ (iOS 14.5+)
- Edge ✓

**Note:** Internet Explorer n'est pas supporté.

## Conseils d'Utilisation

1. **Clarté** - Parlez clairement et à un rythme normal
2. **Bruit ambiant** - Réduisez le bruit de fond pour une meilleure reconnaissance
3. **Langue** - Les commandes sont principalement en français, mais certaines en anglais sont acceptées
4. **Microphone** - Vérifiez que votre microphone est activé et autorisé dans le navigateur
5. **Confidentialité** - Les commandes vocales sont traitées localement sur votre appareil

## Dépannage

### "Je n'ai pas compris cette commande"
- Dites plus clairement
- Réduisez le bruit de fond
- Essayez une formulation légèrement différente
- Dites "aide" pour la liste des commandes

### Le système n'écoute pas
- Vérifiez que le pipeline est en cours d'exécution (status = "running")
- Cliquez sur l'indicateur vocal pour vérifier son état
- Vérifiez les permissions d'accès au microphone dans votre navigateur
- Actualisez la page et réessayez

### Pas d'audio lors de la lecture de pages
- Vérifiez que le volume de votre système n'est pas muet
- Vérifiez que la synthèse vocale est activée dans votre navigateur
- Essayez de dire "à propos" ou une autre page

### Le microphone n'est pas autorisé
- Dans Chrome/Edge : Cliquez sur l'icône du cadenas → Paramètres du site → Microphone → Autorisé
- Dans Firefox : Cliquez sur l'icône du cadenas → Paramètres du site → Microphone → Autoriser

## Architecture Technique

Le système vocal utilise :
- **Web Speech API** pour la reconnaissance vocale (SpeechRecognition)
- **Web Speech API** pour la synthèse vocale (SpeechSynthesis)
- Traitement côté client (aucune donnée ne quitte votre appareil)
- Prise en charge multilingue (français et anglais)

## Fichiers Impliqués

- `interface/js/voice-commands.js` - Module de reconnaissance et synthèse vocales
- `interface/html/index.html` - Page principale avec indicateur vocal
- `interface/html/about.html` - Page À Propos
- `interface/html/contact.html` - Page Contact
- `interface/html/closedones.html` - Page Entourage

## Améliorations Futures

Les améliorations suivantes sont envisagées :
- Support de commandes plus complexes (ex: "ajoute Marie à l'entourage")
- Contrôle vocal des paramètres (ex: "augmente la luminosité de 20%")
- Commandes vocales personnalisées
- Retours auditifs pour chaque action
- Support de langues supplémentaires

---

**Version:** 1.0
**Dernière mise à jour:** Mai 2026
