# Grupa A – jakość danych

## Wasze zadanie
Oceńcie, czy otrzymane dane są wystarczająco poprawne i spójne, aby rozpocząć modelowanie. `y = yes` oznacza założenie lokaty terminowej.

## Zbadajcie
- ile `NaN` i `unknown` jest w każdej zmiennej oraz ile jest wierszy dotkniętych co najmniej jednym z tych problemów;
- przykładowe wiersze z brakami dla wybranej zmiennej (`ZMIENNA`);
- które wartości liczbowe są niemożliwe, a które tylko rzadkie i skrajne; najpierw zobaczycie krótką listę wartości, potem przykładowe pełne wiersze;
- wszystkie kandydaty dla wybranej zmiennej (`ZMIENNA_ANOMALII`);
- czy wartość `pdays = 999` jest błędem;
- jaki odsetek klientów z kategorią `housing` lub `contact` założył lokatę;
- jak średnia i mediana zmieniają rozkład wieku oraz czy klienci z brakującym wiekiem częściej zakładali lokatę niż osoby w wieku 35–45 lat.

## Decyzje o anomaliach
W komórce `DECYZJE` wpiszcie decyzję dla pary `(numer_wiersza, "zmienna")`, np. `"zostaw"` albo `"usuń"`. Ustawcie własny identyfikator `OCENIAJACY`, aby wynik każdej osoby lub grupy zapisał się osobno do `data/decyzje_anomalie_<OCENIAJACY>.csv`. Decyzje są zapisywane do porównania; notebook nie usuwa automatycznie rekordów.

Wartości niemożliwe i rzadkie skrajności to kandydaci do oceny, a nie gotowa odpowiedź. Sprawdźcie kontekst w pełnym wierszu przed podjęciem decyzji.

## Pytania końcowe
1. Czy można usunąć wszystkie wiersze z `NaN` lub `unknown`?
2. Które anomalie są błędami, a które mogą być prawidłowe?
3. Czy kategorie `housing` i `contact` różnią się odsetkiem założonych lokat? Pamiętajcie, że jest to porównanie, nie dowód przyczynowości.
4. Czy klienci z brakującym wiekiem różnią się wynikiem `y` od osób w wieku 35–45 lat?
5. Jakie decyzje o czyszczeniu wymagają wiedzy biznesowej?

Przygotujcie **2–3 wnioski** do przedstawienia całej grupie.
