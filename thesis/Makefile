PROJNAME=main
BUILDDIR=build

.PHONY: $(PROJNAME).pdf all clean

all: $(PROJNAME).pdf

$(PROJNAME).pdf: $(PROJNAME).tex
	mkdir -p $(BUILDDIR)
# 	latexmk -pdflatex=lualatex -pdf -synctex=1 -verbose -interaction=nonstopmode -file-line-error -output-directory=$(BUILDDIR) $<
	latexmk -pdflua main.tex -synctex=1 -shell-escape -interaction=nonstopmode -file-line-error -output-directory=$(BUILDDIR) $<

clean:
	latexmk -output-directory=$(BUILDDIR) -c && rm -rf $(BUILDDIR)/*