# Compilation Guide

This project is built using a provided `Makefile` and is configured to be compiled with the LuaLaTeX engine.

## Prerequisites

To successfully compile this document, you will need the following software and tools installed on your system:

* LaTeX Distribution: A comprehensive distribution like `texlive-full`. The project relies on a wide array of packages, including `tikz`, `pgfplots`, `tcolorbox`, `siunitx`, `listings`, and `acro`.
* LuaLaTeX Engine
* Biber: Required for bibliography processing. The project uses `biblatex` with the Biber backend to manage citations from a `main.bib` file.
* GNU Make: Needed to execute the commands defined in the `Makefile`.
* External Dependencies (Shell Escape): The compilation uses the `-shell-escape` flag. Because the document loads packages like `svg` and `epstopdf`, you will need **Inkscape** and **Ghostscript** installed and added to your system's PATH to convert vector graphics on the fly.

## Compilation Instructions

* **To build the document:**

  ```bash
  make
  ```

  *or*

  ```bash
  make all
  ```

This command will generate the output file `main.pdf` inside a newly created `build/` directory. The `latexmk` tool will automatically run the necessary passes for bibliography compilation and cross-referencing.

* **To clean up auxiliary files:**

  ```bash
  make clean
  ```

This will securely remove the `build/` directory and clear out temporary files generated during compilation to keep your working tree clean.

## Original template

* <https://github.com/ctu-mrs/thesis_template/>
