<img src="decaf_logo.jpg" alt="decaf Logo">

# Automated Recognition and Classification of Trading Cards

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

This repository contains the hardware design, software implementation and thesis
for an automated system designed to recognize and classify trading cards
(specifically Magic: The Gathering).

The project was developed to design and build a simple robot capable of feeding
trading cards one at a time under a camera. The captured images are then
processed using computer vision and machine learning techniques to identify and
classify each card.

## Project Overview

The system is divided into two main functional parts:

1. **Hardware:** An automated card-feeding mechanism driven by stepper motor
and controlled via an Arduino board.
2. **Software:** A Python-based recognition pipeline using machine learning
(ArcFace) to identify the scanned cards based on a constructed dataset.

## Repository Structure

* **[`/hw`](hw/)** — hardware specifications. Contains photos, wiring diagrams
and the final selection of components for the robot.
* **[`/sw`](sw/)** — source code for the project. Includes Arduino firmware for
the robotic feeder and Python scripts for the machine learning and image
processing pipeline.
* **[`/scripts`](scripts/)** — various utility scripts, including tools for
downloading and preparing image datasets.
* **[`/sources`](sources/)** — collected research materials, links to scientific
articles (papers) and tutorials on relevant algorithms.
* **[`/thesis`](thesis/)** — LaTeX source files of the thesis document.
* **[`/presentation`](presentation/)** — slides and materials used for the
thesis presentation/defense.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
