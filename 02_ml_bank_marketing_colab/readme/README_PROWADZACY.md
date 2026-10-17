# Klucz prowadzącego – warsztat ML

## Założenia wspólne
- Każdy notebook niezależnie przygotowuje tę samą próbkę stratyfikowaną 2000 rekordów z UCI.
- Zmienna `y = yes` oznacza założenie lokaty terminowej.
- `duration` nie występuje w danych modelowych: jest znana dopiero po rozmowie i powodowałaby leakage w scenariuszu predykcji przed rozmową.
- `pdays = 999` oznacza brak wcześniejszego kontaktu. Przy przygotowaniu modelu zamieniamy go na brak w `pdays` i tworzymy wskaźnik `had_previous_contact`.
- Test stanowi stałe 20% próbki i jest dzielony stratyfikowanie. Każdy model i każdy wariant cech ocenia się na tych samych wierszach testowych.
- Wielkości 100, 300, 800 i 1500 tworzą zagnieżdżone podzbiory treningowe. Trzy powtórzenia zmieniają losowanie próbek, ale nie test.
- `basic` używa `age` (wiek), `job` (zawód), `education` (wykształcenie).
- `standard` dodaje `housing` (kredyt mieszkaniowy), `contact` (kanał), `month` (miesiąc), `campaign` (liczba kontaktów), `pdays` (dni od wcześniejszego kontaktu) i `had_previous_contact` (wskaźnik z `pdays = 999`).
- `degraded` używa dokładnie tych samych cech co `standard`, ale losowo ustawia brak w 15% komórek cech wybranego podzbioru treningowego. Nie usuwa wierszy ani kolumn; test nie jest degradowany.
- Pipeline dopasowuje imputację, skalowanie (dla Logistic Regression) i one-hot encoding wyłącznie na danych treningowych.
- `duration` jest pomijane we wszystkich wariantach, bo w scenariuszu predykcji przed rozmową nie jest dostępne. Surowe `pdays = 999` jest zamienione na brak w `pdays` i wskaźnik `had_previous_contact = 0`; inne rekordy otrzymują `had_previous_contact = 1`.

## Oczekiwane obserwacje
- Próbki 100 rekordów mogą dawać zmienny recall lokaty (`y = yes`) między powtórzeniami.
- Więcej danych często stabilizuje wynik, ale poprawa nie musi być równa przy każdym zwiększeniu próbki.
- `standard` może przewyższyć `basic`, ale przyrost zależy od modelu i losowania.
- `degraded` pokazuje koszt utraty informacji, bez zmiany testu.
- Baseline „zawsze brak lokaty” może mieć wysoką accuracy, lecz recall lokaty wynosi 0.
- Współczynniki i importance opisują zachowanie konkretnego modelu, nie dowodzą przyczynowości.
- Porównując trzy grupy, pamiętajcie, że zmienia się algorytm; porównanie wariantów wewnątrz każdej grupy najlepiej izoluje wpływ danych.

## Pliki wynikowe
Każdy notebook zapisuje tabelę średnich i odchyleń standardowych w `data/results/`. Wspólne porównanie grup można wykonać po zebraniu trzech plików `*_summary.csv`.
