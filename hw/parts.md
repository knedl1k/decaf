- Venturi generator
    - jaký tlak zvládne kompresor
    - stačí nízký sací průtok, máme krátké hadičky a malou přísavku
    - dostatečnou úroveň vakua - máme papírové karty, jsou ale porézní, neustále se bude muset kompenzovat přisávání malého vzduchu. Stačí nízká nebo střední úroveň
    - nejspíše zbytečný (kolik l/min vzduchu spotřebuje generátor)
    - nějaký generátory už mají v sobě integrovaný ventil i vacuum break na uvolnění karet, to se může hodit

- Suction cup
    - materiál silikon, měkký a dobře přilne
    - měchová přísavka (bellows cup), protože kompenzuje úhlové nedostatky ramena
    - 5 až 10 mm průměr

- Pneumatic ventil 3/2
    - solenoidový, aby se dal ovládat Arduinem
    - napětí podle zdroje, arduino ho samo neutáhne, bude potřeba ho spínat relé / MOSFETem
    - Normally Closed typ, aby bez proudu byl defaultně zavřený.
    - velikost závitů musí odpovídat hadičkám a generátoru, nejspíš M5 nebo 1/8"
    - díky poréznosti karty bude stačit vypnout ventil, karta pak sama spadne

- Linear actuator (osa Z)
    - co třeba hobby servo? MG90S
        - není potřeba driver ani enkodér, stejně jen vyrovnáváme výškový rozdíl
    - nebo třeba pneumatický píst?
        - už máme kompresor
        - byl by potřeba třeba 5/2 ventil pro obousměrný chod a škrtíci ventily pro regulaci rychlosti
        - několik dalších komponent
    - pokud bude mít stepper, tak to bude sice přesný, ale zbytečný overkill nejspíš

- hadičky
    - materiál PU
    - podle ventilu, generátoru, přísavce...
    - fitinky 