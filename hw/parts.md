# Seznam součástek

## Již máme
- kompresor
    - [Sparmax TC-610H Plus](https://www.sparmaxair.com/de/compressor-legend-1/tc-610h-plus)
    - maximální tlak: 60 psi (4.1 bar)
    - průtok vzduchu: 23-28 l/min
    - velikost závitu: 1/8" PS
- zdroj napětí
- Arduino
- stepper + enkodér + kontroler motoru
- exhaust pro venturi generátor

## Potřeba dokoupit
- Venturi generator
    - https://www.tme.eu/cz/katalog/prislusenstvi-pro-pneumatiku_118130/?params=1995%3A1500167_typ-prislusenstvi-pro-pneumatiku%3Azasuvny-generator-vakua
    - jaký tlak zvládne kompresor
    - stačí nízký sací průtok, máme krátké hadičky a malou přísavku
    - dostatečnou úroveň vakua - máme papírové karty, jsou ale porézní, neustále se bude muset kompenzovat přisávání malého vzduchu. Stačí nízká nebo střední úroveň
    - nejspíše zbytečný (kolik l/min vzduchu spotřebuje generátor)
    - nějaký generátory už mají v sobě integrovaný ventil i vacuum break na uvolnění karet, to se může hodit

- suction cup
    - silikon (VMQ); Si-HD je heavy duty, to je zbytečnost
    - tvrdost Shore chceme co nejměkčí
    - průměr 10 mm 
    - měchová přísavka (bellows cup), protože kompenzuje úhlové nedostatky ramena; 1.5 měchu
    - nezapomenout na upevnění pro přísavku

- pneumatic ventil 3/2
    - https://www.tme.eu/cz/katalog/ventily-a-rozdelovace_118125/?params=1996%3A1455528%3B2711%3A1698168%3B98%3A1439012&queryPhrase=ventil
    - solenoidový, aby se dal ovládat Arduinem
    - napětí podle zdroje, Arduino ho samo neutáhne, bude potřeba ho spínat relé / MOSFETem
    - Normally Closed typ, aby bez proudu byl defaultně zavřený.
    - velikost závitů musí odpovídat hadičkám a generátoru
    - díky poréznosti karty bude stačit vypnout ventil, karta pak sama spadne

- hadičky
    - materiál PU
    - podle ventilu, generátoru, přísavce...
    - fitinky 

- linear actuator (osa Z)
    - co třeba hobby servo? MG90S
        - není potřeba driver ani enkodér, stejně jen vyrovnáváme výškový rozdíl
    - nebo třeba pneumatický píst?
        - už máme kompresor
        - byl by potřeba třeba 5/2 ventil pro obousměrný chod a škrtíci ventily pro regulaci rychlosti
        - několik dalších komponent
    - pokud bude mít stepper, tak to bude sice přesný, ale zbytečný overkill nejspíš
