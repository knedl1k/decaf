## celkový koncept
- rozdělení na dvě fáze
    - trénink
    - inference

## train.py
- Adam nebo SGD?

## model.py
- nemáme nikde záruku, že theta + margin bude <= 180 st.
    - jestli budou problémy, tak to je nejspíš tohle
- https://github.com/dlpazs/Face/blob/master/ArcFace.md
- https://medium.com/@payyavulasaiprakash/arcface-loss-function-for-deep-face-recognition-e1ff5e173b52
- https://en.wikipedia.org/wiki/Logit


### trénování
- na vstupu budou karty a jejich ID, což je v našem případě jejich label
- jako backbone (taky feature extractor) bude třeba ResNet, který embeddne obrázky a udělá z nich vektory
    - ResNet by mohl být dobrý, protože už bude umět poznat hrany a tvary
- jako head bude ArcFace, který spočítá úhly mezi vektory; doufejme, že dost dobře
- loss function pak udělá korekce vah
- při trénování dynamicky generování syntetického datasetu, ať zbytečně neplníme disk něčím, co není potřeba



### inference
- máme naučený backbone
- pošleme neznámou fotku, ze které se embeddne vektor, který následně porovnáme s databází
    - porovnání třeba pomocí Cosine Similarity
- PyTorch
- TIMM na image modely? 
    - ResNet, EfficientNet...
- `Albumentations` na augmentace
- CUDA pro grafiku, na CPU se těch výsledních 100k fotek nedá

## Augmentace dat
- Musíme napodobit nekvalitní fotografie karet, abychom se přiblížii reálné situaci
- Syntetická generace dat / Augmentace
- Transformace
    - perspektivní deformace
    - rotace a ořez
    - simulace osvětlení a odlesků
    - rozostření
    - změna pozadí
- `Albumentations`
    - https://pypi.org/project/albumentations/#i-want-to-know-how-to-use-albumentations-with-deep-learning-frameworks

## ArcFace
- pouze loss funkce
- jako backbone je třeba ResNet50, která z obrázků udělá vektor čísel (tzv. embedding)
- u MTG je problém, že velmi často a pravidelně vychází velké množství nových karet. U klasických řešení by bylo potřeba neustále přetrenovávat celou síť. Tady stačí uložit vektory nových karet do databáze a síť je na ně připravena. 
- zároveň ArcFace umí skvěle rozlišit detaily a malé rozdíly na kartách, což je zde esenciální, protože se karty mohou lišit třeba jen v barvě obrázku, anebo jen ikoně edice
- jako backbone buď klasický ResNet100, případně MobileNet (možné rozšíření na mobily?)
- vstup třeba 112 x 112 x 3
    - šířka x výška x hloubka barev
    - protože nám záleží na barvách, musíme i tu řešit
    - 3 protože RGB
- budeme potřebovat spíš větší detail, takže 224x224x3
- karty nejsou čtverce, co s nimi?
    - buď deformovat, nebo doplnit černou barvou
    - prozatím radši budu cílit na padding, protože deformace by mohla zmasakrovat text a jiné detaily, pak můžu vyzkoušet
- výstup vektor délky 512
