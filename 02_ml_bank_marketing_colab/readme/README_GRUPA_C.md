# Grupa C – Random Forest

## Wasze zadanie
Sprawdźcie, jak Random Forest wykorzystuje dane o różnej wielkości i jakości oraz które wejściowe informacje są najbardziej pomocne przy odnajdywaniu klientów, którzy założą lokatę.

## Uruchomcie
1. Tabelę i wykres wyników dla `basic`, `standard` i `degraded` oraz rozmiarów treningu 100, 300, 800 i 1500.
2. Tabelę permutation importance: jak zmienia się recall lokaty (`y = yes`) po przetasowaniu jednej cechy w teście?
3. Wizualizację jednego, płytkiego drzewa z lasu.

Wyniki modeli są porównywalne między grupami: wspólne rozmiary próbek, ten sam podział train/test i ten sam zestaw scenariuszy. `duration` nie jest używane.

## Co oznaczają zestawy danych?
- `basic`: `age` (wiek), `job` (zawód), `education` (wykształcenie).
- `standard`: wszystkie przygotowane cechy dostępne w scenariuszu przed rozmową: `age`, `job`, `education`, `housing` (kredyt mieszkaniowy), `contact` (kanał), `month` (miesiąc), `campaign` (liczba kontaktów), `pdays` (dni od wcześniejszego kontaktu) i `had_previous_contact` (wskaźnik wcześniejszego kontaktu).
- `degraded`: cechy są takie same jak w `standard`, ale 15% ich komórek w treningu jest losowo zamienianych na braki; test nie jest zmieniany.

Zestaw `degraded` nie oznacza mniejszej liczby rekordów. Braki obsługuje pipeline: mediana dla liczb, najczęstsza wartość dla kategorii. `pdays = 999` jest kodowane jako brak w `pdays` i wartość `had_previous_contact = 0`; `duration` jest wyłączone ze wszystkich wariantów.

## Pytania
- Czy ranking cech jest taki sam dla różnych modeli?
- Czy cecha ważna dla modelu oznacza, że powoduje założenie lokaty?
- Jak zmienia się stabilność wyników przy 100 i 1500 przykładach?
- Czy brak części wartości (`degraded`) przesuwa ranking lub wynik?

Zapiszcie podsumowanie z `data/results/` do wspólnego porównania grup.
