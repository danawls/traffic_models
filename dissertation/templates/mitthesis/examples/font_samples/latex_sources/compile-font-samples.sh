#! /bin/bash

# first run

pdflatex Defaultfonts
pdflatex Fira_Newtxsf
pdflatex Libertine
pdflatex Lucida
pdflatex Newtx-sans-text
pdflatex Newtx

lualatex Heros-Stix2
lualatex Stix2
lualatex Termes-stix2
lualatex Termes

# biber

biber Defaultfonts
biber Fira_Newtxsf
biber Libertine
biber Lucida
biber Newtx-sans-text
biber Newtx
biber Heros-Stix2
biber Stix2
biber Termes-stix2
biber Termes

# second run

pdflatex Defaultfonts
pdflatex Fira_Newtxsf
pdflatex Libertine
pdflatex Lucida
pdflatex Newtx-sans-text
pdflatex Newtx

lualatex Heros-Stix2
lualatex Stix2
lualatex Termes-stix2
lualatex Termes

# third run

pdflatex Defaultfonts
pdflatex Fira_Newtxsf
pdflatex Libertine
pdflatex Lucida
pdflatex Newtx-sans-text
pdflatex Newtx

lualatex Heros-Stix2
lualatex Stix2
lualatex Termes-stix2
lualatex Termes

# clean up

mkdir pdffiles
mv *.pdf pdffiles
rm *.aux
rm *.bbl
rm *.bcf
rm *.blg
rm *.log
rm *.lot
rm *.lof
rm *.toc
rm *.xml


