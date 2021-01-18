# Fitness App Ai

## Tim
- Veljko Plećaš SW68-2017
- Ivana Marošević SW74-2017
- Petar Cerović SW26-2017

## Asistenti
- Dragan Vidaković

## Definicija problema
Cilj je da se detektuje količina nutritivnih vrednosti na slici. Potrebno je da se na slici detektuje svo voće i povrće koje se nalazi na njemu, odraditi njihovu klasifikaciju, i na osnovu rezultata i tablice o količini kalorija svakog voća/povrća odrediti ukupnu količinu nutritivnih vrednosti sa slike.

## Motivacija
Rešenje ovog problema može koristiti svim sportistima i ljudima koji vode računa o svojoj ishrani.

## Skup podataka
Za treniranje i testiranje neuronske mreže koristićemo [KAGGLE](https://www.kaggle.com/moltean/fruits) skup podataka. Po potrebi ćemo priložiti još neke ručno sakupljene slike za validaciju.

## Metodologija
Za klasifikaciju podataka će se raditi treniranje konvolucionih neuronskih mreža nad relevantnim skupom podataka
Detekcija objekata na slici će se raditi upotrebom adaptivnog treshold-a

## Evaluacija
Za evaluaciju detekcije objekata koristićemo intersection over union uz pomoć koje ćemo videti koliko su detektovane regije precizne. Evaluacija klasifikacije objekata biće urađenja uz pomoć accuracy metrike (ili cross entropy loss) koja će nam određivati preciznost klasifikacije. Za rezultate nutritivnih vrednosti koristićemo Mean squared error metriku.
