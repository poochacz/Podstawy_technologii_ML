# Grupa A – Logistic Regression

## Wasze zadanie
Sprawdźcie, jak zmienia się wynik Logistic Regression, gdy zmieniamy liczbę przykładów treningowych i ilość informacji dostępnych modelowi.

## Uruchomcie
1. Tabelę i wykres dla `basic`, `standard` i `degraded` oraz rozmiarów treningu 100, 300, 800 i 1500.
2. Powtórzenia dla kilku różnych próbek treningowych. Zwróćcie uwagę na średni wynik i jego zmienność.
3. Wykres współczynników Logistic Regression dla klasy `yes`.

Notebook przygotowuje dane, podział train/test i przetwarzanie kolumn automatycznie. Nie zmieniajcie testu ani nie usuwajcie ręcznie rekordów. `duration` jest celowo wyłączone.

## Co oznaczają zestawy danych?
- `basic`: `age` (wiek), `job` (zawód), `education` (wykształcenie) — tylko podstawowe informacje.
- `standard`: powyższe oraz `housing` (kredyt mieszkaniowy), `contact` (kanał kontaktu), `month` (miesiąc), `campaign` (liczba kontaktów), `pdays` (dni od wcześniejszego kontaktu) i `had_previous_contact` (czy wcześniejszy kontakt wystąpił).
- `degraded`: ten sam zestaw cech co `standard`, ale 15% komórek z cechami w treningu jest losowo ukrywanych jako braki. Nie usuwa to rekordów, a test pozostaje niezmieniony.

Wariant `degraded` pokazuje wpływ jakości informacji, a nie wpływ mniejszej liczby cech. Pipeline automatycznie uzupełnia braki medianą dla liczb i najczęstszą wartością dla kategorii. `pdays = 999` jest zamieniane na brak w `pdays` oraz zachowywane jako wskaźnik `had_previous_contact`. `duration` jest pomijane we wszystkich zestawach, bo w tym scenariuszu nie jest znane przed rozmową.

## Pytania
- Czy większa próbka zawsze daje ten sam przyrost jakości?
- Czy wyniki dla 100 rekordów są stabilne między powtórzeniami?
- Co oznacza dodatni lub ujemny współczynnik? Czy oznacza przyczynowość?
- Jak `degraded` wypada względem `standard`?

Zapiszcie podsumowanie z `data/results/` do wspólnego porównania grup.
