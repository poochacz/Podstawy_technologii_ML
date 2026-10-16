# Grupa B – niezbalansowane klasy i data leakage

## Wasze zadanie
Sprawdźcie, czy sam wynik `accuracy` wystarczy do oceny przyszłego modelu, jak cechy różnią klasy `yes` i `no` oraz czy wszystkie silnie związane z wynikiem cechy wolno wykorzystać. `y = yes` oznacza założenie lokaty terminowej.

## Zbadajcie
- jaki jest udział klas `yes` i `no`;
- jaką accuracy uzyskałby „model”, który **zawsze przewiduje najczęstszą klasę**;
- tabelę różnic dla wszystkich cech: średnie dla liczbowych oraz najbardziej różniący się udział kategorii dla kategorycznych;
- wykres jednej wybranej cechy (`CECHA`), liczbowej albo kategorycznej;
- liczbę `yes` i `no` oraz odsetek lokat **wewnątrz każdej kategorii** `job`, `education`, `housing`, `contact` i `month`, porównany z wynikiem całej próbki;
- szczególnie przyjrzyjcie się `duration` i sprawdźcie, **kiedy dana cecha jest dostępna** względem momentu predykcji.

## Kontekst
Predykcję chcemy wykonywać **przed rozpoczęciem rozmowy z klientem**.

## Pytania końcowe
1. Czy accuracy będzie wystarczającą metryką?
2. Jaki baseline należy przyjąć?
3. Które cechy warto wykorzystać w modelu przewidującym wynik **przed rozmową**, a które mogą powodować leakage?
4. Czy kategoria z wysokim odsetkiem lokat zawsze oznacza, że cecha powoduje założenie lokaty?

Przygotujcie **2–3 wnioski** do przedstawienia całej grupie.
