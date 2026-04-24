import pyttsx3

engine = pyttsx3.init()

def say(text: str):
    """Dit le texte à voix haute."""
    engine.say(text)
    engine.runAndWait()

def set_voice(lang: str = "fr"):
    """Change la langue : 'fr' ou 'en'."""
    voices = engine.getProperty('voices')
    for voice in voices:
        if lang == "fr" and "french" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            return
        if lang == "en" and "english" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            return

def set_speed(rate: int = 150):
    """Vitesse de parole. Défaut=150, lent=100, rapide=200."""
    engine.setProperty('rate', rate)

def set_volume(vol: float = 1.0):
    """Volume entre 0.0 et 1.0."""
    engine.setProperty('volume', vol)

def list_voices():
    """Affiche toutes les voix disponibles."""
    voices = engine.getProperty('voices')
    for i, voice in enumerate(voices):
        print(f"[{i}] {voice.name} — {voice.id}")

if __name__ == "__main__":
    list_voices()
    set_speed(150)
    set_volume(10.0)
    say("Bonjour, je suis ton assistant vocal.")