from app import db

# Crează toate tabelele definite în `app.py`
with db.app.app_context():
    db.create_all()

print("Baza de date a fost creată cu succes!")
