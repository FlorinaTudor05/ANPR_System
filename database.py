from app import app, db  # Importă atât aplicația Flask cât și baza de date

# Inițializează contextul Flask pentru a crea tabelele
with app.app_context():
    db.create_all()

print("Baza de date a fost creată cu succes!")
