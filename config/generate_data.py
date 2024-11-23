import pandas as pd
import numpy as np
import random
 
# Ustawienie seeda dla powtarzalności wyników
np.random.seed(42)
 
# Przygotowanie danych
n_records = 20
 
# Przygotowanie list wartości dla zmiennych kategorycznych
wyksztalcenie_options = ["podstawowe", "średnie", "wyższe", "zawodowe"]
plec_options = ["K", "M"]
stanowisko_options = ["Specjalista", "Kierownik", "Dyrektor", "Asystent", "Konsultant"]
departament_options = ["IT", "HR", "Finanse", "Marketing", "Sprzedaż", "Produkcja"]
 
# Generowanie danych
data = {
    "dochod": np.random.normal(6000, 2000, n_records).round(2),  # Średnia 6000, odchylenie 2000
    "wiek": np.random.randint(25, 60, n_records),
    "wyksztalcenie": random.choices(wyksztalcenie_options, k=n_records),
    "plec": random.choices(plec_options, k=n_records),
    "stanowisko": random.choices(stanowisko_options, k=n_records),
    "staz_pracy": np.random.randint(1, 30, n_records),
    "departament": random.choices(departament_options, k=n_records)
}
 
# Korekta dochodu na podstawie stanowiska i stażu pracy
for i in range(n_records):
    # Modyfikacja dochodu na podstawie stanowiska
    if data["stanowisko"][i] == "Dyrektor":
        data["dochod"][i] *= 2.5
    elif data["stanowisko"][i] == "Kierownik":
        data["dochod"][i] *= 1.5
    elif data["stanowisko"][i] == "Asystent":
        data["dochod"][i] *= 0.7
    
    # Dodatek za staż pracy (1% za rok)
    data["dochod"][i] *= (1 + data["staz_pracy"][i] * 0.01)
    
    # Zaokrąglenie dochodu
    data["dochod"][i] = round(data["dochod"][i], 2)
 
# Utworzenie DataFrame
df = pd.DataFrame(data)
 
# Zapewnienie, że staż pracy jest logiczny względem wieku
df["staz_pracy"] = df.apply(lambda row: min(row["staz_pracy"], row["wiek"] - 18), axis=1)
 
# Sortowanie po departamencie i stanowisku dla lepszej czytelności
df = df.sort_values(["departament", "stanowisko"])
 
# Zapisanie do CSV
df.to_csv("sample_data.csv", index=False)
 
# Wyświetlenie danych
print(df.to_string())
