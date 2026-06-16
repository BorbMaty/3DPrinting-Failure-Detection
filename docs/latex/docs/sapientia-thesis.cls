\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sapientia-thesis}[2026/05/06 Sapientia diploma thesis template]

\LoadClass[12pt,a4paper,oneside]{report}

% ── Encoding and typography ──────────────────────────────────────────────────
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[hungarian,provide=*,english]{babel}
\RequirePackage{mathptmx}   % Times New Roman clone
\RequirePackage{microtype}

% ── Page layout ─────────────────────────────────────────────────────────────
% Word template: left 3 cm, right 2.5 cm, top 2.5 cm, bottom 2.5 cm
\RequirePackage[a4paper,
  left=3cm, right=2.5cm, top=2.5cm, bottom=2.5cm,
  headsep=14pt]{geometry}

\RequirePackage{setspace}
\setstretch{1.15}
\setlength{\parindent}{1.25cm}
\setlength{\parskip}{0pt}

% ── Utilities ────────────────────────────────────────────────────────────────
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{longtable}
\RequirePackage{caption}
\RequirePackage{float}
\RequirePackage{enumitem}
\RequirePackage{etoolbox}
\RequirePackage{xcolor}
\RequirePackage{fancyhdr}
\RequirePackage{tocloft}
\RequirePackage{amsmath}
\RequirePackage[hidelinks]{hyperref}
% NOTE: this repo builds with the classic bibtex toolchain (ieeetr style) both
% locally and in CI (compileLatex.yml runs `bibtex main`). biblatex/biber are
% not installed here, so the original biblatex setup is intentionally omitted.
% Use \bibliographystyle{ieeetr} + \bibliography{...} in the document instead.

% ── Code listings ───────────────────────────────────────────────────────────
\RequirePackage{listings}
\lstset{
  basicstyle=\small\ttfamily,
  keywordstyle=\color{blue},
  commentstyle=\color{gray},
  stringstyle=\color[rgb]{0.6,0,0},
  breaklines=true,
  frame=none,
  tabsize=2,
  showstringspaces=false,
  xleftmargin=1.25cm,   % align with body indent
}

% ── Figure / table caption style ────────────────────────────────────────────
% Word: "N. ábra: Title" — italic label, italic text, centered
\captionsetup{font=small, labelfont=it, textfont=it,
              labelsep=colon, justification=centering}

% ── Header / footer ─────────────────────────────────────────────────────────
% Word body pages: thin rule in header (no text), page number centred in footer
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{}
\fancyhead[R]{}
\fancyfoot[C]{\thepage}

% Chapter opening pages use plain style → same look
\fancypagestyle{plain}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
  \fancyfoot[C]{\thepage}
}

% Title / abstract pages: no header rule, no page number printed in footer
% (page counter continues but is suppressed via \pagenumbering{gobble})
\fancypagestyle{titlestyle}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

% ── TOC typography ──────────────────────────────────────────────────────────
% Word TOC: chapter entries bold, section/subsection normal weight
\setlength{\cftbeforechapskip}{0.25em}
\setlength{\cftbeforesecskip}{0em}
\renewcommand{\cftchapfont}{\bfseries}
\renewcommand{\cftchappagefont}{\bfseries}
\renewcommand{\cftsecfont}{\normalfont}
\renewcommand{\cftsecpagefont}{\normalfont}
\renewcommand{\cftsubsecfont}{\normalfont}
\renewcommand{\cftsubsecpagefont}{\normalfont}

% TOC title: centred, bold, all-caps  (e.g. "TARTALOMJEGYZÉK")
\renewcommand{\cfttoctitlefont}{\hfill\Large\bfseries\MakeUppercase}
\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftloftitlefont}{\hfill\Large\bfseries\MakeUppercase}
\renewcommand{\cftafterloftitle}{\hfill}
\renewcommand{\cftlottitlefont}{\hfill\Large\bfseries\MakeUppercase}
\renewcommand{\cftafterlottitle}{\hfill}

% ── TOC with dots for chapters ───────────────────────────────────────────────
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}  % Add dots for chapters
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}   % Ensure sections have dots (already default)

% ── Numbering depth ──────────────────────────────────────────────────────────
\setcounter{secnumdepth}{2}  % number down to subsection only (e.g. 5.2.1)
\setcounter{tocdepth}{2}

% ── Bibliography file helper ─────────────────────────────────────────────────
% bibtex variant: just remember the file name; the document calls
% \bibliographystyle / \bibliography itself.
\newcommand{\thesisbibfile}[1]{\def\@thesisbibfile{#1}}

\makeatletter

% ── Metadata commands ────────────────────────────────────────────────────────
\newcommand{\universityro}[1]{\def\@universityro{#1}}
\newcommand{\facultyro}[1]{\def\@facultyro{#1}}
\newcommand{\programro}[1]{\def\@programro{#1}}
\newcommand{\titleRO}[1]{\def\@titleRO{#1}}
\newcommand{\coordlabelro}[1]{\def\@coordlabelro{#1}}
\newcommand{\supervisorRO}[1]{\def\@supervisorRO{#1}}
\newcommand{\studentlabelro}[1]{\def\@studentlabelro{#1}}

\newcommand{\universityhu}[1]{\def\@universityhu{#1}}
\newcommand{\facultyhu}[1]{\def\@facultyhu{#1}}
\newcommand{\programhu}[1]{\def\@programhu{#1}}
\newcommand{\titleHU}[1]{\def\@titleHU{#1}}
\newcommand{\supervisorHU}[1]{\def\@supervisorHU{#1}}

\newcommand{\universityen}[1]{\def\@universityen{#1}}
\newcommand{\facultyen}[1]{\def\@facultyen{#1}}
\newcommand{\programen}[1]{\def\@programen{#1}}
\newcommand{\titleEN}[1]{\def\@titleEN{#1}}
\newcommand{\supervisorEN}[1]{\def\@supervisorEN{#1}}

\newcommand{\studentname}[1]{\def\@studentname{#1}}
\newcommand{\yearofthesis}[1]{\def\@yearofthesis{#1}}

% ── Sensible defaults ────────────────────────────────────────────────────────
\universityro{Universitatea „Sapientia" din Cluj-Napoca}
\facultyro{FACULTATEA DE ȘTIINȚE TEHNICE ȘI UMANISTE,\\TÎRGU MUREȘ}
\programro{SPECIALIZAREA AUTOMATICĂ ȘI INFORMATICĂ APLICATĂ}
\titleRO{TITLU LUCRARE}
\coordlabelro{Coordonator științific}
\supervisorRO{Nume conducător temă\\Ex.:\\\c{S}.l.dr.ing. Fehér Áron\\Conf.dr.ing. Brassai Sándor Tihamér\\Prof.dr.ing. Dávid László}
\studentlabelro{Absolvent}

\universityhu{Sapientia Erdélyi Magyar Tudományegyetem}
\facultyhu{MAROSVÁSÁRHELYI KAR}
\programhu{AUTOMATIKA ÉS ALKALMAZOTT INFORMATIKA SZAK}
\titleHU{Dolgozat címe}
\supervisorHU{Témavezető neve és fokozatai\\Példák tudományos fokozatra:\\Dr. Fehér Áron, egyetemi adjunktus\\Dr. Brassai Sándor Tihamér, egyetemi előadótanár\\Dr. Dávid László, egyetemi tanár}

\universityen{Sapientia Hungarian University of Transylvania}
\facultyen{FACULTY OF TECHNICAL AND HUMAN SCIENCES,\\TARGU MURES}
\programen{AUTOMATION AND APPLIED INFORMATICS}
\titleEN{Thesis Title}
\supervisorEN{Supervisor name and academic title\\Examples:\\Lecturer Dr. Fehér Áron\\Assoc. Prof. Dr. Brassai Sándor Tihamér\\Prof. Dr. Dávid László}

\studentname{Diák neve}
\yearofthesis{2026}

% ── Romanian cover page ──────────────────────────────────────────────────────
% Visual (from Word):
%   • University: italic (not bold), centred
%   • Faculty + Program: bold, centred
%   • Large vertical gap
%   • Thesis title: very large, NOT bold, centred
%   • "PROIECT DE DIPLOMĂ": large, NOT bold, centred
%   • vfill → supervisor block left / student block right
%   • Year: bold, centred
\newcommand{\maketitlepagero}{%
  \begin{titlepage}
    \thispagestyle{titlestyle}
    \centering
    {\Large\bfseries \@universityro\par}
    \vspace{0.3cm}
    {\large\bfseries \@facultyro\par}
    \vspace{0.2cm}
    {\large\bfseries \@programro\par}
    \vspace{3.5cm}
    {\fontsize{24}{28}\selectfont \@titleRO\par}
    \vspace{2.5cm}
    {\fontsize{18}{22}\selectfont PROIECT DE DIPLOMĂ\par}
    
    \vfill % Pushes the names toward the bottom

    % 1. Supervisor (Aligned Left)
    \begin{flushleft}
      {\large \@coordlabelro:}\\[-0.2cm]
      {\large\bfseries \@supervisorRO\par}
    \end{flushleft}

    \vspace{1.0cm} % 2. The Clearance (Adjust to 1.5cm if you have fewer supervisors)

    % 3. Student (Pushed to the Right)
    \begin{flushright}
      \begin{minipage}{0.42\textwidth}
        \raggedright % Keeps text inside the box left-aligned
        {\large \@studentlabelro:}\\[-0.2cm]
        {\large\bfseries \@studentname\par}
      \end{minipage}
    \end{flushright}
    
    \vfill
    {\large\bfseries \@yearofthesis\par}
  \end{titlepage}
  \clearpage % Force a clean break before abstracts
}


% ── Hungarian cover page ─────────────────────────────────────────────────────
% Visual:
%   • University: bold, centred
%   • Faculty + Program: bold, centred
%   • Thesis title: large, NOT bold, centred
%   • "Diplomadolgozat": large, NOT bold, centred
%   • supervisor left / student right
%   • Year: NOT bold, centred
\newcommand{\maketitlepagehu}{%
  \begin{titlepage}
    \thispagestyle{titlestyle}
    \centering
    {\large\bfseries \@universityhu\par}
    \vspace{0.3cm}
    {\large\bfseries \@facultyhu\par}
    \vspace{0.2cm}
    {\large\bfseries \@programhu\par}
    \vspace{3.5cm}
    {\fontsize{24}{28}\selectfont \@titleHU\par}
    \vspace{2.5cm}
    {\fontsize{18}{22}\selectfont Diplomadolgozat\par}
    
    \vfill
    
    \begin{flushleft}
      {\large Témavezető:}\\[-0.2cm]
      {\large\bfseries \@supervisorHU\par}
    \end{flushleft}

    \vspace{1.0cm}

    \begin{flushright}
      \begin{minipage}{0.42\textwidth}
        \raggedright
        {\large Végzős hallgató:}\\[-0.2cm]
        {\large\bfseries \@studentname\par}
      \end{minipage}
    \end{flushright}
    
    \vfill
    {\large\bfseries \@yearofthesis\par}
  \end{titlepage}
  \clearpage
}

% ── English cover page ───────────────────────────────────────────────────────
\newcommand{\maketitlepageen}{%
  \begin{titlepage}
    \thispagestyle{titlestyle}
    \centering
    {\large\bfseries \@universityen\par}
    \vspace{0.3cm}
    {\large\bfseries \@facultyen\par}
    \vspace{0.2cm}
    {\large\bfseries \@programen\par}
    \vspace{3.5cm}
    {\fontsize{24}{28}\selectfont \@titleEN\par}
    \vspace{2.5cm}
    {\fontsize{18}{22}\selectfont Diploma Thesis\par}
    
    \vfill
    
    \begin{flushleft}
      {\large Supervisor:}\\[-0.2cm]
      {\large\bfseries \@supervisorEN\par}
    \end{flushleft}

    \vspace{1.0cm}

    \begin{flushright}
      \begin{minipage}{0.45\textwidth}
        \raggedright
        {\large Graduate:}\\[-0.2cm]
        {\large\bfseries \@studentname\par}
      \end{minipage}
    \end{flushright}
    
    \vfill
    {{\large\bfseries \@yearofthesis\par}}
  \end{titlepage}
  \clearpage
}
% ── Abstract page ────────────────────────────────────────────────────────────
% Usage: \abstractpage{thesisTitle}{headingLabel}{bodyText}{keywordsLine}
%
% Word layout (e.g. page 2):
%   "Dolgozat címe románul"  ← thesis title, large, NOT bold, centred
%   "Extras"                 ← heading, large, NOT bold, centred
%   body text, justified, 1.5 line-spacing
%   "Kulcsszavak: ..."       ← bold-italic label + normal text
%
% \abstractpage{Label}{body}{keywords}
% Call \abstractpage with: heading label, body, keywords
% The thesis title is shown automatically from \@titleXX.
%
% To support different title per language, we use an optional first argument:
% \abstractpagero, \abstractpagehu, \abstractpageen
\newcommand{\abstractpagero}[3]{%   {heading}{body}{keywords}
  \clearpage\thispagestyle{plain}%
  \begin{center}
    {\fontsize{18}{22}\selectfont \@titleRO\par}
    \vspace{0.8cm}
    {\fontsize{18}{22}\selectfont #1\par}
  \end{center}
  \vspace{1.0cm}
  \begingroup\setstretch{1.5}\noindent #2\par\endgroup
  \vspace{0.8cm}
  \noindent #3\par
}

\newcommand{\abstractpagehu}[3]{%
  \clearpage\thispagestyle{plain}%
  \begin{center}
    {\fontsize{18}{22}\selectfont \@titleHU\par}
    \vspace{0.8cm}
    {\fontsize{18}{22}\selectfont #1\par}
  \end{center}
  \vspace{1.0cm}
  \begingroup\setstretch{1.5}\noindent #2\par\endgroup
  \vspace{0.8cm}
  \noindent #3\par
}

\newcommand{\abstractpageen}[3]{%
  \clearpage\thispagestyle{plain}%
  \begin{center}
    {\fontsize{18}{22}\selectfont \@titleEN\par}
    \vspace{0.8cm}
    {\fontsize{18}{22}\selectfont #1\par}
  \end{center}
  \vspace{1.0cm}
  \begingroup\setstretch{1.5}\noindent #2\par\endgroup
  \vspace{0.8cm}
  \noindent #3\par
}

% Generic fallback (kept for backward compat, title must be passed manually)
\newcommand{\abstractpage}[3]{%
  \clearpage\thispagestyle{plain}%
  \begin{center}
    {\fontsize{18}{22}\selectfont #1\par}
  \end{center}
  \vspace{1.0cm}
  \begingroup\setstretch{1.5}\noindent #2\par\endgroup
  \vspace{0.8cm}
  \noindent #3\par
}

% ── Hungarian names ──────────────────────────────────────────────────────────
\addto\captionshungarian{%
  \renewcommand{\contentsname}{Tartalomjegyzék}%
  \renewcommand{\listfigurename}{Ábrák jegyzéke}%
  \renewcommand{\listtablename}{Táblázatok jegyzéke}%
}

% ── Appendix with continued numeric numbering ────────────────────────────────
\makeatletter
\renewcommand{\appendix}{%
  % Don't reset chapter counter, just continue numbering
  \renewcommand{\thechapter}{\arabic{chapter}}%
  \renewcommand{\thesection}{\thechapter.\arabic{section}}%
  \renewcommand{\thesubsection}{\thesection.\arabic{subsection}}%
}

% ── Appendix with continued numeric numbering ────────────────────────────────
\makeatletter
\renewcommand{\appendix}{%
  % Don't reset chapter counter, just continue numbering
  \renewcommand{\thechapter}{\arabic{chapter}}%
  \renewcommand{\thesection}{\thechapter.\arabic{section}}%
  \renewcommand{\thesubsection}{\thesection.\arabic{subsection}}%
}

\makeatother