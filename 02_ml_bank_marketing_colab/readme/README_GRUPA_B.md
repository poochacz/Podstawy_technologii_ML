# Grupa B – Decision Tree

## Wasze zadanie
Sprawdźcie, jak liczba i jakość danych wpływa na wynik drzewa decyzyjnego oraz jak można prześledzić pojedynczą decyzję.

## Uruchomcie
1. Tabelę i wykres wyników dla `basic`, `standard` i `degraded` oraz rozmiarów treningu 100, 300, 800 i 1500.
2. Wizualizację drzewa ograniczoną do czytelnej głębokości.
3. Ścieżkę decyzji dla jednego przykładowego klienta z testu.

Pipeline automatycznie uzupełnia braki, koduje kategorie i trenuje model. Nie trzeba przygotowywać danych ręcznie. `duration` jest wyłączone, bo przed rozmową nie jest znane.

## Co oznaczają zestawy danych?
- `basic`: `age` (wiek), `job` (zawód), `education` (wykształcenie).
- `standard`: te trzy cechy oraz `housing` (kredyt mieszkaniowy), `contact` (kanał kontaktu), `month` (miesiąc), `campaign` (liczba kontaktów), `pdays` (dni od wcześniejszego kontaktu) i `had_previous_contact` (czy wcześniejszy kontakt wystąpił).
- `degraded`: te same cechy co `standard`, z 15% losowo ukrytych komórek w danych treningowych. Rekordy nie są usuwane, a test pozostaje bez zmian.

Porównanie `basic`/`standard` dotyczy ilości dostępnych cech; `standard`/`degraded` — kompletności ich wartości. Braki są automatycznie uzupełniane wewnątrz pipeline’u: medianą dla liczb i najczęstszą wartością dla kategorii. `pdays = 999` jest kodowane osobnym wskaźnikiem `had_previous_contact`; `duration` nie wchodzi do żadnego zestawu.

## Pytania
- Jak zmienia się recall lokaty (`y = yes`) wraz z liczbą rekordów?
- Czy dwie próbki tej samej wielkości dają identyczny wynik?
- Jakie warunki doprowadziły drzewo do decyzji dla przykładowego klienta?
- Czy bardziej szczegółowy podział automatycznie oznacza lepszą decyzję biznesową?

Zapiszcie podsumowanie z `data/results/` do wspólnego porównania grup.
