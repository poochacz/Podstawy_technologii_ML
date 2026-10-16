# Klucz prowadzącego – EDA

## Wspólne
Próbka ma 2000 rekordów i 10 kolumn:
`age`, `job`, `education`, `housing`, `contact`, `month`, `duration`, `campaign`, `pdays`, `y`.

Próbka jest celowo lekko niereprezentatywna względem pełnego `bank-additional-full.csv`.
Dodano także niewielką liczbę kontrolowanych problemów jakości danych.

## Grupa A – oczekiwane obserwacje
- rzeczywiste `NaN`: `age`, `education`, `housing`;
- w oryginalnym UCI występuje również kategoria `unknown`;
- kontrolowane wartości błędne: m.in. wiek `160`, `-4`, ujemne `campaign`, `duration`, `pdays`;
- `pdays=999` wygląda jak outlier, ale w UCI oznacza brak wcześniejszego kontaktu;
- duże wartości `campaign`, `duration`, wiek itp. nie muszą być błędami;
- `job`, `education`, `housing`, `contact`, `month` są kategoryczne;
- średnia może silniej reagować na skrajne wartości niż mediana.

## Grupa B – oczekiwane obserwacje
- `y` jest niezbalansowane: klasa `no` wyraźnie dominuje;
- baseline „zawsze no” daje wysoką accuracy, więc samo accuracy jest słabym punktem odniesienia;
- `duration` bardzo mocno różni `yes` i `no`;
- zgodnie z opisem UCI jest to czas ostatniego kontaktu, więc nie jest znany przed zakończeniem rozmowy;
- przy predykcji przed rozmową użycie `duration` jest data leakage.

## Grupa C – oczekiwane obserwacje
Próbka została pobrana wagowo:
- `may` jest nadreprezentowany;
- `mar`, `sep`, `oct`, `dec` są niedoreprezentowane;
- `cellular` jest lekko nadreprezentowany;
- klienci >60 są niedoreprezentowani.
Uczestnicy powinni zauważyć różnice na wykresach/tabelach bez znajomości mechanizmu próbkowania.

## Uwaga dydaktyczna
Nie zdradzaj od razu, które dane są sztucznie uszkodzone. Najważniejsza jest rozmowa:
„Czy to błąd, rzadki realny przypadek, kod specjalny, czy cecha nieznana w momencie predykcji?”
