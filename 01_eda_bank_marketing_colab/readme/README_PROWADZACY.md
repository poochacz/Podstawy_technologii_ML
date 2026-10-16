# Klucz prowadzącego – EDA

## Wspólne
Próbka ma 2000 rekordów i 10 kolumn:
`age`, `job`, `education`, `housing`, `contact`, `month`, `duration`, `campaign`, `pdays`, `y`.

Próbka jest celowo lekko niereprezentatywna względem pełnego `bank-additional-full.csv`.
`reference` jest pełnym zbiorem źródłowym ograniczonym do tych samych 10 kolumn. Metadane warsztatowe zachowują listę kolumn oryginalnego pliku i tych pominiętych w warsztacie.

Dodano także niewielką liczbę kontrolowanych problemów jakości danych. `y = yes` oznacza założenie lokaty terminowej; nie jest informacją o przyznaniu kredytu.

## Grupa A – oczekiwane obserwacje
- kontrolowane `NaN`: po 18 w `age`, 24 w `education`, 14 w `housing` oraz po 8 w `month`, `duration`, `campaign`, `pdays`;
- kontrolowane tekstowe `unknown`: 8 w `month` oraz po 4 w `duration`, `campaign`, `pdays`; w oryginalnych kategoriach UCI `job`, `education`, `housing` również występuje `unknown`;
- pierwsze podsumowanie pokazuje liczbę braków per kolumna i unikalną liczbę wierszy dotkniętych `NaN` lub `unknown`; przykłady dla wybranej zmiennej są próbkowane, nie stanowią pełnego wykazu;
- kontrolowane wartości błędne: m.in. wiek `160`, `-4`, ujemne `campaign`, `duration`, `pdays`;
- użytkownicy oceniają kandydatów na anomalie: lista wartości jest krótka, pełne wiersze i widok per zmienna są osobno; decyzje `zostaw`/`usuń` zapisują się do pliku CSV wraz z identyfikatorem oceniającego, bez automatycznego usuwania danych;
- `pdays=999` wygląda jak outlier, ale w UCI oznacza brak wcześniejszego kontaktu;
- skrajna wartość nie musi być błędem; lista kandydatów nie rozstrzyga decyzji za uczestnika;
- `job`, `education`, `housing`, `contact`, `month` są kategoryczne;
- porównanie kategorii `housing` i `contact` pokazuje odsetek klientów z lokatą w każdej kategorii;
- analiza wieku porównuje korelację z `y` oraz wyniki dla brakującego wieku i osób w wieku 35–45 lat;
- imputacja średnią lub medianą zmienia statystyki wieku, ale nie zmienia targetu `y`.

## Grupa B – oczekiwane obserwacje
- `y` jest niezbalansowane: klasa `no` wyraźnie dominuje;
- baseline „zawsze no” daje wysoką accuracy, więc samo accuracy jest słabym punktem odniesienia;
- tabela porównuje wszystkie cechy; dla liczbowych pokazuje średnie, dla kategorycznych udział najbardziej różniącej się kategorii w klasach `no` i `yes`;
- osobny wykres jest wybierany przez `CECHA`; może pokazać cechę liczbową lub kategoryczną;
- dodatkowa tabela liczy `brak lokaty`/`lokata` i odsetek lokat wewnątrz kategorii, z ogólnym odsetkiem jako punktem odniesienia;
- `duration` bardzo mocno różni `yes` i `no`;
- zgodnie z opisem UCI jest to czas ostatniego kontaktu, więc nie jest znany przed zakończeniem rozmowy;
- przy predykcji przed rozmową użycie `duration` jest data leakage.

## Grupa C – oczekiwane obserwacje
Próbka została pobrana wagowo:
- `may` jest nadreprezentowany;
- `mar`, `sep`, `oct`, `dec` są niedoreprezentowane;
- `cellular` jest lekko nadreprezentowany;
- klienci >60 są niedoreprezentowani.
Uczestnicy powinni zauważyć różnice na wykresach/tabelach bez znajomości mechanizmu próbkowania. Notebook dodaje też:
- pokrycie kolumn źródłowych: lista pokazuje, które z oryginalnych kolumn zachowano w zestawie warsztatowym;
- kategorie obecne w referencji, ale nieobecne w próbce, z liczebnością i udziałem w referencji;
- odsetek `y = yes` w segmentach `month`, `contact`, `job` porównany między próbką i referencją; małe grupy należy interpretować ostrożnie;
- symulację wykluczenia kategorii (domyślnie `job = admin.`) tylko z próbki; referencja pozostaje bez zmian, co pokazuje wpływ decyzji na liczebność i odsetek lokat.

## Uwaga dydaktyczna
Nie zdradzaj od razu, które dane są sztucznie uszkodzone. Najważniejsza jest rozmowa:
„Czy to błąd, rzadki realny przypadek, kod specjalny, czy cecha nieznana w momencie predykcji?”
