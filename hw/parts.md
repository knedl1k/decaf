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
    - https://cz.rs-online.com/web/p/vyvevy/2315358
    - průměr trysky 0.5 mm --- nízká spotřeba vzduchu
    - kompresor dává 23-28 l/min, ejektor asi 15-20 l/min, takže ideál
    - má tlumič
    - výkon je naprosto dostačující
    - vnější průměr hadiček 6mm
    <!-- - https://www.tme.eu/cz/katalog/prislusenstvi-pro-pneumatiku_118130/?params=1995%3A1500167_typ-prislusenstvi-pro-pneumatiku%3Azasuvny-generator-vakua -->
    - poznámky:
        - jaký tlak zvládne kompresor
        - stačí nízký sací průtok, máme krátké hadičky a malou přísavku
        - dostatečnou úroveň vakua - máme papírové karty, jsou ale porézní, neustále se bude muset kompenzovat přisávání malého vzduchu. Stačí nízká nebo střední úroveň
        - nejspíše zbytečný --- kolik l/min vzduchu spotřebuje generátor
        - nějaký generátory už mají v sobě integrovaný ventil i vacuum break na uvolnění karet, to se může hodit

- suction cup
    - https://cz.rs-online.com/web/p/pneumaticke-prisavky/2038181
    - možná je moc tvrdý?
    - možná moc velký průměr? 
    - poznámky:
        - silikon (VMQ); Si-HD je heavy duty, to je zbytečnost
        - tvrdost Shore chceme co nejměkčí
        - průměr ? mm 
        - měchová přísavka (bellows cup), protože kompenzuje úhlové nedostatky ramena; 1.5 měchu
        - nezapomenout na upevnění pro přísavku

- pneumatic ventil 3/2
    - https://cz.rs-online.com/web/p/pneumaticke-solenoidoveridici-regulacni-ventily/2337369?gb=a
    - to jsou fakt takhle drahý??
    - nejspíš VK334, protože má Rc 1/8" závity, takže dá se nástrčné šroubení G 1/8" pro 6mm hadičku
        - VK332 má menší porty, takže by se hůř hledaly fitinky
    - podle datasheetu VK333-XG, kde X je:
        - 5 když 24 VDC
        - 6 když 12 VDC
        - G znamená Groomet s kabely, to bude lepší pro Arduino jak DIN?
    - ještě potřeba:
        - 2 fitinky pro vstupní a pracovní port
        - 1 tlumič na výfuk?
        - spínač
    <!-- - https://www.tme.eu/cz/katalog/ventily-a-rozdelovace_118125/?params=1996%3A1455528%3B2711%3A1698168%3B98%3A1439012&queryPhrase=ventil -->
    - poznámky:
        - solenoidový, aby se dal ovládat Arduinem
        - napětí podle zdroje, Arduino ho samo neutáhne, bude potřeba ho spínat relé / MOSFETem
        - Normally Closed typ, aby bez proudu byl defaultně zavřený
        - velikost závitů musí odpovídat hadičkám a generátoru

- hadičky
    - https://cz.rs-online.com/web/p/vzduchove-hadice/1745728
    - ofc nepotřebujem tak dlouhý, tohle je jen pro ukázku
    - vnější průměr 6mm
    - "nástrčné šroubení G 1/8" (vnitřní) pro 6mm hadičku
    - poznámky:
        - materiál PU
        - podle ventilu, generátoru, přísavce...
        - fitinky

- linear actuator (osa Z)
    - https://www.gme.cz/vysledky-vyhledavani?q=servomotor
    - co třeba hobby servo? MG90S
        - není potřeba driver ani enkodér, stejně jen vyrovnáváme výškový rozdíl
    - nebo třeba pneumatický píst?
        - už máme kompresor
        - byl by potřeba třeba 5/2 ventil pro obousměrný chod a škrtíci ventily pro regulaci rychlosti
        - několik dalších komponent
    - pokud bude mít stepper, tak to bude sice přesný, ale zbytečný overkill
