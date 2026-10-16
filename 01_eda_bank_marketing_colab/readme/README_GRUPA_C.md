# Grupa C – bias i reprezentatywność

## Wasze zadanie
Sprawdźcie, czy 2000 rekordów używanych podczas warsztatu wygląda podobnie do szerszego zbioru referencyjnego UCI.

## Zbadajcie
Porównajcie próbkę warsztatową z wybranym zbiorem referencyjnym pod względem:
- miesiąca kontaktu (`month`);
- kanału kontaktu (`contact`);
- zawodu (`job`);
- struktury wieku;
- udziału klasy `yes`;
- kategorii obecnych w referencji, których brakuje w próbce;
- odsetka założonych lokat w obrębie kategorii `month`, `contact` i `job` w próbce oraz referencji.

Notebook pokazuje też, które kolumny z oryginalnego pliku UCI zostały pominięte w zestawie warsztatowym. To informacja o zakresie danych: próbka i bieżąca referencja zawierają te same wybrane kolumny.

W punkcie 7 możecie zasymulować wykluczenie kategorii, np. zawodu `admin.`. Zmienia się tylko próbka warsztatowa; pełna referencja zostaje bez zmian. Porównajcie liczebność i odsetek lokat przed i po wykluczeniu oraz różnicę względem referencji.

## Pytania końcowe
1. Które kolumny źródłowe pominięto i czy któraś z nich mogłaby być ważna biznesowo?
2. Czy próbce brakuje kategorii istotnych dla klientów?
3. W których segmentach odsetek założonych lokat najbardziej różni się od referencji?
4. Co zmieniło wykluczenie wybranej kategorii? Czy próbka stała się bardziej, czy mniej podobna do referencji?
5. Jak nad- lub niedoreprezentowanie grup może wpłynąć na model i jak pobralibyście lepszą próbkę?

## Ważne
Pełny zbiór UCI jest tutaj **zbiorem referencyjnym**, a nie idealnym opisem całej populacji klientów banku. To również jest ważne ograniczenie analizy reprezentatywności.

Przygotujcie **2–3 wnioski** do przedstawienia całej grupie.
