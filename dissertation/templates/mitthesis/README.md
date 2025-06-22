  
  #mitthesis --- A LaTeX template for an MIT thesis#

  v1.14 dated 2024/07/19

  ####Overview####
  This class provides a LaTeX template to format an MIT thesis according to
  the requirements of the Massachusetts Institute of Technology Libraries (as posted in 2024):
  
  [https://libraries.mit.edu/distinctive-collections/thesis-specs/](https://libraries.mit.edu/distinctive-collections/thesis-specs/)

  This template is appropriate for MIT theses of all types.
  
  This template works with either pdfLaTeX or unicode engines such as luaLaTeX. The bibliography can be prepared with either biblatex (default) or natbib/bibtex. The class is based on current LaTeX distributions, ideally 11/2022 or later, but  compatible with distributions back to 2020. This template replaces the older version of mitthesis.cls, which was first composed in the 1980s.
  
  With minor adjustments, this template can be adapted for use at other institutions (see the documentation for details).

  The files in this distribution are:

          README.md             --  this file
          mitthesis.cls         --  the class file
          MIT-Thesis.pdf        --  a sample thesis from the template, using default fonts
          *
          MIT-thesis-template/  --  a directory with the files needed to starting writing your thesis
              MIT-Thesis.tex        --  the main latex template file for this class
              abstract.tex          --  put your abstract in this file
              acknowledgments.tex   --  put your acknowledgments in this file
              biosketch.tex         --  put your biosketch in this file (optional)
              chapter1.tex          --  put your first chapter in this file (etc.)
              appendixa.tex         --  put your first appendix in this file (etc.)
              mitthesis-sample.bib  --  a sample bibliography file with many examples
              mydesign.tex          --  an optional file to load packages for document design
              fontsets/             --  a subdirectory of input files that load optional fonts
          *
          mitthesis-doc/        --  documentation for usage and options
          examples/font_samples/  
                                --  sample theses in different fonts 
          examples/cover_page_samples/  
                                --  sample theses for one or more authors and degrees
          
    
  ####Author####
  
  John H. Lienhard V
  
  Department of Mechanical Engineering
          
  Massachusetts Institute of Technology
          
  Cambridge, MA 02139-4307 USA


 ---
 
 ####Change log####
 v1.14 (19 July 2024)
 -  add backward compatibility code for \\text_titlecase_all:n
 -  adjust supervisor and acceptor titles used on sample cover pages
 -  format J/psi as \\symbfit in sample chapter 1 and regenerate font samples
 -  edit documentation
 
 v1.13 (03 July 2024)
 -  same as v1.12, except now includes the correct documentation
 
 v1.12 (02 July 2024)
 -  add logic for one degree issued by two departments
 -  fix missing space in abstract block for multiple departments
 -  adjust second department layout on title and abstract pages
 -  table of contents revised
 -  revise documentation
 -  switch default citation style to numeric (from IEEE).  Provide examples for IEEE and author/year styles.
 - 	add \\AtEndPreamble{..} to mitthesis-newtx-sans-text.tex to accommodate v1.731 of newtx
 -  in chapter1.tex, replace \\text by \\textrm; fix nested link in section heading
 -  remove obsolete hyperref option from xcolor
 -  minor changes to log notes of fontset files
 -  code clean up
 
 v1.11 (02 November 2023)
 - revise all skips on cover page to better group material while allowing for glue compression as content increases; increase font size of author name; provide user macros for control cover page spacing and author name font.
 - use \\mdseries for linenumbers in all cases, rather than using the locally active series
 - remove \\raggedright in favor of \\bibsetup for bibliography in MIT-Thesis.tex 
 - add backward compatibility for alt tag of \\includegraphics for pre-2021/11/15 distributions, add alt tag in chapter1.tex
 - edit all occurrences of \\addcontentsline
 - rearrange eqn:WT1 in chapter1.tex to avoid margin overflow with some fonts, eliminate associated work-around.
 - edit tab:1 and increase space below caption; edit fig:golden
 - edit nomenclature environment to better accommodate [future] tagged pdf
 - remove vertical [1em] after final \\Acceptor (thanks to Gustav Pettersson)
 - edit documentation

 v1.10 (23 September 2023)
 - Minor edit of documentation

 v1.09 (22 September 2023)
 - Revise documentation
 - Minor code clean-up
 
 v1.08 (11 September 2023):
 - accommodate recent changes in hyperxmp package (used when \\DocumentMetadata is not called)
 - various minor edits
 
 v1.07 (04 September 2023):
 - Improve handling of custom fontset files; move fontset directory into MIT-thesis-template directory
 - Remove unnecessary latex code
 - Move hyperlink and line number color choices to mydesign.tex
 - Enable some backward compatibility in expl3 code
 - Edit documentation
  
 v1.06 (29 August 2023):
 - Extensive revision of class file, with most code converted to expl3.  Significant streamlining of remaining LaTeX2e code. Minor changes to user interface. No changes to format or design of thesis.
 - Add user macros \\CopyrightAuthor, \\DegreeMonth, \\DegreeYear, and \\Institution{..}
 - Drop user macros \\CCurl, \\PDFRightsText, and \\MIT{..}
 - Reduce volume of pdf metadata required to be added by user
 - Accommodate author names that end with a period or include a comma (e.g., Martin Luther King, Jr.) or end in a capital letter (e.g., King Charles III).
 - Reduce font size in fira-newtxsf fontset
 - Update documentation
 
 v1.05 (21 July 2023): 
 - Fix bug in toc page number affecting some lists of figures or tables (note: for backward compatibility, remove code in .tex file around \\tableofcontents, \\listoffigures, and \\listoftables so that your code will match the present version)
 - Include thesis submission date in abstract, remove previous degrees from abstract page, and allow May for degree month, per MIT Libraries
 - Fill pdflicenseurl in class file; update a class warning message
 - Clean-up documentation and code

 v1.04 (3 July 2023): 
 - Embed default fontset in class file, in case fontset directory is missing

 v1.03 (26 June 2023): 
 - Bug fix: fontset naming for older LaTeX formats
 - Change default biblatex style to IEEE
 - Code hacks for non-MIT use of template (see documentation, pg. 7)

 v1.02 (23 June 2023): 
 - Bug fixes: triple major counter, \\SignatureBlockSize
 - Code revisions for cover page and abstract pages: spacing, linebreaking, and user command options
  
 v1.01 (19 June 2023): Changes to file structure and naming
 
 v1.00 (17 June 2023): Initial release
 
 ---
 
 ####License####

 Copyright (c) 2023 John H. Lienhard

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
 associated documentation files (the "Software"), to deal in the Software without restriction, 
 including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
 subject to the following two conditions:

 The above copyright notice and this permission notice shall be included in all copies or 
 substantial portions of the Software.

 The software is provided "as is", without warranty of any kind, express or implied, including but 
 not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. 
 In no event shall the authors or copyright holders be liable for any claim, damages or other liability, 
 whether in an action of contract, tort or otherwise, arising from, out of or in connection with the 
 software or the use or other dealings in the software.
