# Warsztat EDA – UCI Bank Marketing

## Cel
Pracujemy na jednym problemie: **przed rozpoczęciem kontaktu z klientem chcemy ocenić, czy klient założy lokatę terminową**.

Źródło danych: UCI Machine Learning Repository, **Bank Marketing**. W warsztacie używamy wybranych 10 kolumn oraz próbki 2000 rekordów. Pole `y = yes` oznacza założenie lokaty, a nie przyznanie kredytu.

## Organizacja
1. Uruchom notebook `00_EDA_wspolne.ipynb`, aby przygotować próbkę i zbiór referencyjny.
2. Po części wspólnej przejdź do notebooka swojej grupy:
   - Grupa A – braki danych, anomalie, kategorie i imputacja wieku;
   - Grupa B – niezbalansowane klasy, porównanie cech i ryzyko data leakage;
   - Grupa C – reprezentatywność, pominięte kategorie i wpływ wykluczenia segmentu.

## Praca z notebookami
Nie musicie pisać kodu analitycznego; gotowe funkcje są w katalogu `scripts/`. W niektórych komórkach wybieracie jednak zakres analizy, np. `ZMIENNA`, `CECHA` albo kategorię do symulacji. W Grupie A można wpisać własne decyzje `zostaw`/`usuń` dla anomalii; każda osoba lub grupa zapisuje osobny plik CSV.

Wyniki są materiałem do interpretacji i dyskusji. Różnice w próbie nie dowodzą same w sobie zależności przyczynowej, a pełny zbiór UCI jest punktem odniesienia, nie idealnym obrazem wszystkich klientów banku.
