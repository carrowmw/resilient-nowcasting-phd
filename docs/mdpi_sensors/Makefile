# Define the name of the main LaTeX file without the .tex extension
MAIN = mdpi_sensors

# Define the PDF viewer command
PDFVIEW = open

# The default target is 'all'
all: $(MAIN).pdf

# Rule to build the PDF from the LaTeX file using LaTeX directly
$(MAIN).pdf: $(MAIN).tex
	export PATH=/Library/TeX/texbin:$$PATH && pdflatex -interaction=nonstopmode $(MAIN).tex
	export PATH=/Library/TeX/texbin:$$PATH && bibtex $(MAIN)
	export PATH=/Library/TeX/texbin:$$PATH && pdflatex -interaction=nonstopmode $(MAIN).tex
	export PATH=/Library/TeX/texbin:$$PATH && pdflatex -interaction=nonstopmode $(MAIN).tex

# Rule to clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.blg *.dvi *.fdb_latexmk *.fls *.lof *.lot *.synctex.gz

# Rule to open the PDF with the default PDF viewer
view: $(MAIN).pdf
	$(PDFVIEW) $(MAIN).pdf

.PHONY: all clean view
