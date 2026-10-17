# Warsztat ML – UCI Bank Marketing

## Cel
Zobaczyć, jak liczba i jakość informacji wpływają na przewidywanie, czy klient założy lokatę terminową. To warsztat używania gotowych modeli, nie programowania ani strojenia hiperparametrów.

`y = yes` oznacza założenie lokaty. `duration` nie jest używane: w scenariuszu predykcji przed rozmową ta wartość nie jest jeszcze znana.

## Uruchomienie lokalne
1. Otwórz folder repozytorium w VS Code.
2. Zainstaluj zależności: `pip install -r 02_ml_bank_marketing_colab/requirements.txt`.
3. Otwórz jeden z trzech notebooków i uruchom komórki od góry.

## Uruchomienie w Colabie
Otwórz notebook z repozytorium GitHub i uruchom komórki po kolei. Jeśli repozytorium nie jest dostępne w środowisku Colab, komórka startowa sklonuje je automatycznie. Każdy notebook jest niezależny i sam przygotowuje dane.

Pierwsze uruchomienie wymaga dostępu do Internetu, aby pobrać dane UCI oraz, jeśli potrzeba, pakiety. Dane i wyniki są następnie zapisywane lokalnie w folderze `data/` tego warsztatu.

## Notebooki
- Grupa A – Logistic Regression: wpływ współczynników.
- Grupa B – Decision Tree: uproszczone drzewo i prześledzenie jednej decyzji.
- Grupa C – Random Forest: permutation importance i jedno przykładowe drzewo.

Każda grupa porównuje te same wielkości zbioru treningowego (`100`, `300`, `800`, `1500`) na tym samym stratyfikowanym teście. Notebooki zapisują podsumowania wyników w `data/results/`, aby można je było zestawić po ćwiczeniu.

## Jak czytać eksperyment

Każdy notebook porównuje trzy warianty danych. Są to nazwy scenariuszy, nie trzy różne modele:

| Wariant | Użyte informacje | Co sprawdzamy |
|---|---|---|
| `basic` | `age` (wiek), `job` (zawód), `education` (wykształcenie) | Jak działa model z małym, podstawowym zestawem informacji? |
| `standard` | `age`, `job`, `education`, `housing` (kredyt mieszkaniowy), `contact` (kanał kontaktu), `month` (miesiąc), `campaign` (liczba kontaktów), `pdays` (dni od wcześniejszego kontaktu) i `had_previous_contact` (czy kontakt wcześniej wystąpił) | Wynik przy wykorzystaniu wszystkich przygotowanych cech dostępnych w przyjętym scenariuszu. |
| `degraded` | Te same cechy co w `standard` | Jak wynik zmienia się, gdy losowo 15% komórek z cechami w treningu zostaje ukrytych jako brak danych? |

`degraded` **nie usuwa 15% rekordów ani kolumn** i nie zmienia testu. Pipeline uzupełnia braki: medianą dla cech liczbowych, najczęstszą wartością dla kategorii; kodowanie i imputacja są dopasowywane tylko na treningu. `duration` (czas rozmowy) jest wykluczone ze wszystkich wariantów, bo przed rozmową nie jest znane. W źródle `pdays = 999` oznacza brak wcześniejszego kontaktu; ta informacja jest zachowana w `had_previous_contact`, a samo `pdays` jest traktowane jako brak i uzupełniane w pipeline.

Próbki treningowe są zagnieżdżone. Powtórzenia pokazują, jak wynik waha się przy innym losowaniu rekordów, a nie przy innym teście. Raportowane metryki to accuracy i recall lokaty (`y = yes`); prosty baseline zawsze przewiduje `no`.
