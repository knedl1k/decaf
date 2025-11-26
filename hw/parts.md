# Seznam součástek

## Již máme
- kompresor
    - [Sparmax TC-610H Plus](https://www.sparmaxair.com/de/compressor-legend-1/tc-610h-plus)
    - maximální tlak: 60 psi (4.1 bar)
    - průtok vzduchu: 23-28 l/min
    - velikost závitu: 1/8" PS
- zdroj napětí
- Arduino Mega
- stepper 12-24 V
- enkodér 
    - `AS5147-TS_EK_AB`
- driver motoru 
    - `TMC2209`
- servo motor

## Vyrobíme na 3D tiskárně
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

## Potřeba dokoupit
**KV** je zkratka pro kompresory-vzduchotechnika\
**rs** je zkratka pro rs-online
- suction cup
    - [KV: 11 mm průměr](https://www.kompresory-vzduchotechnika.cz/prisavka-sb-11-mm-silikon/)
        - mají i větší, ale 11 by mělo dostačovat <https://www.kompresory-vzduchotechnika.cz/prisavky-sb--s-1-5-vlnovcem/>
        - buď koupit pružinový píst anebo fitinku závit M5 na 6mm hadičku 
    - [rs: 30 mm průměr](https://cz.rs-online.com/web/p/pneumaticke-prisavky/2038181)

- pneumatic ventil 3/2
    - [KV: 155 gramů](https://www.kompresory-vzduchotechnika.cz/solenoidove-ventily-g356-18b-mosaz/)
        - 1.6 mm průtok; varianta A má průtok 1.2 mm
        - 2x G1/8" na 6mm hadičku 
        - 1x M5 pro výfuk
    - [rs: 120 gramů](https://cz.rs-online.com/web/p/pneumaticke-solenoidoveridici-regulacni-ventily/2337369?gb=a)
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
    - [KV: volitelná délka](https://www.kompresory-vzduchotechnika.cz/hadicka-z-polyuretanu-modra-6-4-mm/)
    - [rs: 30 metrů](https://cz.rs-online.com/web/p/vzduchove-hadice/1745728)
    - vnější průměr 6mm
    - PU

- fitinky
    - [KV: M5 na 6mm hadičku](https://www.kompresory-vzduchotechnika.cz/prime-sroubeni-6-m5/)
    - [KV: G1/8" na 6mm hadičku](https://www.kompresory-vzduchotechnika.cz/prime-sroubeni-6-g1-8/)

