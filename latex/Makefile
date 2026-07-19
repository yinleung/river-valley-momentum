# Build the paper with Tectonic (https://tectonic-typesetting.github.io).
#   make         compile main.tex -> main.pdf
#   make clean   remove build outputs
PAPER := main

.PHONY: all clean
all:
	tectonic $(PAPER).tex --synctex

clean:
	rm -f $(PAPER).pdf $(PAPER).synctex.gz $(PAPER).log
