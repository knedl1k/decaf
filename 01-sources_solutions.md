# Trading Card Recognition and Classification: Research Review and Solutions

## Commercial Solutions and Existing Systems

**Card Dealer Pro** [^1] - Relatively expensive, database of 20 million unique card records, featuring auto-cropping. Used for sorting cards intended for sale.

**CardBot** [^2] - Extra pricey (8k USD hw + 1450 USD sw), can scan 1,000 cards per hour and sort based on "any" 
attribute, using silicone suction cups for gentle card handling. Sorting cards intended for sale.

**Magic Sorter** [^3] - Not transparent about the price, csv export of sorted cards with integration to real time market.

**TCG Machines** [^4] - Even less transparent, they claim it works, but who knows. But they have [Foil detection](
https://support.tcgmachines.com/knowledge/foil-detection) where they use several LEDs and diffuser panels to get clear
images.

**ErieTCG OCR** [^5] seems that it cannot be purchased (nor used), but at the same time it is the closest thing to what 
we would like to do. It represents a more specialized approach: "extracting structured data from trading cards with 
YOLO models for object detection and PyTesseract/TrOCR for text extraction. The system employs microservice 
architecture with custom Named Entity Recognition and knowledge graph-based text preprocessing." No clue what that means
but definetly it means something.

## Academic Research Systems

Academic approaches focus on sophisticated computer vision techniques. A notable **Chinese Poker Self-Playing Robot** [^6] demonstrates card handling using TM5-900 robotic arm with custom suction mechanisms and YOLOv5 for card recognition. The system achieves card identification through object detection and employs greedy algorithms for strategic card selection [^6].

<br>

#### By this point I am feeling that YOLO does not mean You Life Only Once and it seems pretty important.

<br>

Research in **ID card verification** [^7] shows relevant methodologies, combining YOLOv8 for segmentation, ResNet-based classification with triplet margin loss and angular margin loss, and OCR technologies (Tesseract, EasyOCR, PaddleOCR). The system achieves 90% segmentation accuracy and demonstrates the effectiveness of metric learning approaches [^7].


[^1]: https://www.carddealerpro.com

[^2]: https://cardcastle.co/cardbot

[^3]: https://www.magic-sorter.com

[^4]: https://tcgmachines.com

[^5]: https://www.oodles.com/our-work/it-software/python/1784

[^6]: https://arxiv.org/html/2312.09455

[^7]: https://repositum.tuwien.at/handle/20.500.12708/213271