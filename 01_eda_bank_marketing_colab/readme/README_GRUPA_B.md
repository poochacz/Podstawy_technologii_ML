# Grupa B – niezbalansowane klasy i data leakage

## Wasze zadanie
Sprawdźcie, czy sam wynik `accuracy` wystarczy do oceny przyszłego modelu oraz czy wszystkie silnie skorelowane z wynikiem cechy wolno wykorzystać.

## Zbadajcie
- jaki jest udział klas `yes` i `no`;
- jaką accuracy uzyskałby „model”, który **zawsze przewiduje najczęstszą klasę**;
- które cechy numeryczne najbardziej różnią klientów `yes` i `no`;
- szczególnie przyjrzyjcie się kolumnie `duration`;
- zastanówcie się, **kiedy dana cecha jest faktycznie dostępna** względem momentu wykonywania predykcji.

## Kontekst
Predykcję chcemy wykonywać **przed rozpoczęciem rozmowy z klientem**.

## Pytania końcowe
1. Czy accuracy będzie wystarczającą metryką?
2. Jaki baseline należy przyjąć?
3. Czy każda bardzo predykcyjna cecha powinna znaleźć się w modelu?
4. Która cecha może powodować leakage i dlaczego?

Przygotujcie **2–3 wnioski** do przedstawienia całej grupie.
