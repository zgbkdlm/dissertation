%% aaltoseries.cls 1.2.1
%% Class file for the Aalto University publication series
%% ---
%% Copyright:
%% Aalto University Publications
%% julkaisut(a)aalto.fi
%%
%% Created and maintained by:
%% Jussi Pekonen
%% Department of Signal Processing and Acoustics
%% Aalto University School of Electrical Engineering
%%
%% Maintained (2017-) by:
%% Martin Vermeer
%% Department of the Built Environment 
%% Aalto University School of Engineering
%%
%% URL: https://wiki.aalto.fi/display/aaltolatex/
%% ---
%% LICENSE
%% The Aalto University series class file has been published under the Creative Commons Attribution license
%% (\url{http://creativecommons.org/licenses/by/1.0/}). This means that you can use the class freely in your own documents.
%% ---
%% Updated 2012/03/11
%%
%% Updated 2017/03/01 Martin Vermeer: bug fixes, thx Jussi Tarvainen
%% Updated 2017/03/15 Martin Vermeer: fouriernc support
%% Updated 2017/03/22 Martin Vermeer: added better vertical layout option, microtype 
%% Updated 2017/12/26 Martin Vermeer: allow for 'oneside' option
%% Updated 2017/12/27 Martin Vermeer: added pdfa option for PDF/A production
%% Updated 2018/01/11 Panu Hiekkataipale (PH): useability improvements
%% Updated 2018/08/07 Martin Vermeer: hyperref links to headers fix
%% Updated 2018/08/08 Martin Vermeer: introduced LoF, LoT, Abbreviations, Symbols with proper hyperref links.

\def\AaltoSeriesVersion{1.2}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aaltoseries}[2017/03/01 v\AaltoSeriesVersion Aalto University series class file]

%% Default class. Allow 'oneside' option
\@ifclasswith{aaltoseries}{oneside}{%
\LoadClass[11pt,oneside]{report}
}{%
\LoadClass[11pt,twoside,openright]{report}
}

%% Required for conditionals
\RequirePackage{ifthen}
% Conditional for an article collection
\newboolean{aaltoseriesarticlecollection}
% Conditional for a dissertation
\newboolean{aaltoseriesdissertation}
% Conditional for a report
\newboolean{aaltoseriesreport}
% Conditional for a teaching material
\newboolean{aaltoseriesteachingmaterial}
% Conditional for a working paper
\newboolean{aaltoseriesworkingpaper}
% Conditional for individual articles
\newboolean{aaltoseriesarticle}
\setboolean{aaltoseriesarticle}{false}
% Conditional for abstracts only
\newboolean{aaltoseriesabstractsonly}
\setboolean{aaltoseriesabstractsonly}{false}
% Conditional for very short publications that do not use chapters
\newboolean{aaltoseriesnochapters}
\setboolean{aaltoseriesnochapters}{false}
% Conditional for the author's contribution setting in the dissertation mode
\newboolean{aaltoseriesdissertationnocontribution}
\setboolean{aaltoseriesdissertationnocontribution}{false}
% Conditional for the page numbering setting in the dissertation mode
\newboolean{aaltoseriesdissertationnopagenumbers}
\setboolean{aaltoseriesdissertationnopagenumbers}{false}
% Conditional for drafting
\newboolean{aaltoseriesdraft}
\setboolean{aaltoseriesdraft}{false}
% Conditional for better looking math
\newboolean{aaltoseriesmath}
\setboolean{aaltoseriesmath}{false}
% Conditional for better top, bottom margins and linespacing (thx Hassan Naseri)
\newboolean{aaltoseriesvertlayout}
\setboolean{aaltoseriesvertlayout}{false}
% Conditional for PDF/A compatibility
\newboolean{pdfa}
\setboolean{pdfa}{false}
% Suppress aaltologo if you want to call your own
\newboolean{nologo}
\setboolean{nologo}{false}

%% Class options
% Option 'articlecollection' - for article collections
\DeclareOption{articlecollection}{%
	\setboolean{aaltoseriesarticlecollection}{true}%
	\setboolean{aaltoseriesdissertation}{false}%
	\setboolean{aaltoseriesreport}{false}%
	\setboolean{aaltoseriesteachingmaterial}{false}%
	\setboolean{aaltoseriesworkingpaper}{false}%
}
%Option  'dissertation' - for dissertations
\DeclareOption{dissertation}{%
	\setboolean{aaltoseriesarticlecollection}{false}%
	\setboolean{aaltoseriesdissertation}{true}%
	\setboolean{aaltoseriesreport}{false}%
	\setboolean{aaltoseriesteachingmaterial}{false}%
	\setboolean{aaltoseriesworkingpaper}{false}%
}
% Option 'report' - for report publications
\DeclareOption{report}{%
	\setboolean{aaltoseriesarticlecollection}{false}%
	\setboolean{aaltoseriesdissertation}{false}%
	\setboolean{aaltoseriesreport}{true}%
	\setboolean{aaltoseriesteachingmaterial}{false}%
	\setboolean{aaltoseriesworkingpaper}{false}%
}
% Option 'teachingmaterial' - for teaching materials
\DeclareOption{teachingmaterial}{%
	\setboolean{aaltoseriesarticlecollection}{false}%
	\setboolean{aaltoseriesdissertation}{false}%
	\setboolean{aaltoseriesreport}{false}%
	\setboolean{aaltoseriesteachingmaterial}{true}%
	\setboolean{aaltoseriesworkingpaper}{false}%
}
% Option 'workingpaper' - for working papers
\DeclareOption{workingpaper}{%
	\setboolean{aaltoseriesarticlecollection}{false}%
	\setboolean{aaltoseriesdissertation}{false}%
	\setboolean{aaltoseriesreport}{false}%
	\setboolean{aaltoseriesteachingmaterial}{false}%
	\setboolean{aaltoseriesworkingpaper}{true}%
}
% Option 'article' - for individual articles of an article collection
\DeclareOption{article}{%
	\setboolean{aaltoseriesarticlecollection}{false}%
	\setboolean{aaltoseriesdissertation}{false}%
	\setboolean{aaltoseriesreport}{false}%
	\setboolean{aaltoseriesteachingmaterial}{false}%
	\setboolean{aaltoseriesworkingpaper}{false}%
	\setboolean{aaltoseriesarticle}{true}%	
}
% Option 'abstracts' - for typesetting special abstracts and only abstracts
\DeclareOption{abstracts}{%
	\setboolean{aaltoseriesabstractsonly}{true}%
}
% Option 'nochapters' - for very short publications
\DeclareOption{nochapters}{%
	\setboolean{aaltoseriesnochapters}{true}%
}
% Option 'nocontribution' - for suppressing the Author's Contribution from the article disserations
\DeclareOption{nocontribution}{%
	\setboolean{aaltoseriesdissertationnocontribution}{true}%
}
% Option 'nopagenumbers' - for suppressing the page numbering from the included articles in the article disserations
\DeclareOption{nopagenumbers}{%
	\setboolean{aaltoseriesdissertationnopagenumbers}{true}%
}
% Option 'draft' - for drafting
\DeclareOption{draft}{%
	\setboolean{aaltoseriesdraft}{true}%
	\PassOptionsToPackage{draft}{pdfpages}%
}
% Option 'draft*' - for drafting with other packages having their non-draft modes on
\DeclareOption{draft*}{%
	\setboolean{aaltoseriesdraft}{true}%
}
% Option 'final' - for the full feature exploitation
\DeclareOption{final}{%
	\setboolean{aaltoseriesdraft}{false}%
}
% Option 'math' - use better looking math font with New Cent Schoolbook
\DeclareOption{math}{%
	\setboolean{aaltoseriesmath}{true}%
}
% Option 'vertlayout' - better looking vertical layout, top, bottom margins
\DeclareOption{vertlayout}{%
	\setboolean{aaltoseriesvertlayout}{true}%
}
% Option 'pdfa' - produce PDF/A compatible file
\DeclareOption{pdfa}{%
	\setboolean{pdfa}{true}%
}
\DeclareOption{nologo}{%
	\setboolean{nologo}{true}
}

% Other options - no processing
\DeclareOption*{}

%% Default options - report
\ExecuteOptions{report}

%% Process options
\ProcessOptions*\relax

%% Check that you use chapters in dissertations and article collections
\ifthenelse{\boolean{aaltoseriesdissertation}}{\setboolean{aaltoseriesnochapters}{false}}{}
\ifthenelse{\boolean{aaltoseriesarticlecollection}}{\setboolean{aaltoseriesnochapters}{true}}{}
\ifthenelse{\boolean{aaltoseriesarticle}}{\setboolean{aaltoseriesnochapters}{true}}{}

%% Set the fonts
\ifthenelse{\boolean{nologo}}
{}{\ifthenelse{\boolean{aaltoseriesdraft}}{\RequirePackage{aaltologo}}{\RequirePackage[Uncoated]{aaltologo}}}

\RequirePackage[protrusion=true, expansion=true,final,babel=true]{microtype}
%% Set math font
\ifthenelse{\boolean{aaltoseriesmath}}{\usepackage{fouriernc2}}{}

% Produce PDF/A
\ifthenelse{\boolean{pdfa}}
{\usepackage[a-1b]{pdfx}}{}

%% Set the marginals
\RequirePackage{geometry}
\newlength{\aaltoseriestmplength}
\setlength{\aaltoseriestmplength}{0.9cm}
\addtolength{\aaltoseriestmplength}{-8pt}
\ifthenelse{\boolean{aaltoseriesabstractsonly}}{%
	% Set the geometry
	\geometry{%
		paperwidth=123mm,%
		paperheight=126mm,%
		textwidth=123mm,%
		textheight=126mm,%
		left=0.4mm,%
		right=0mm,%
		top=0mm,%
		bottom=0mm,%
		ignoreheadfoot,%
		headheight=0mm,%
		headsep=0mm%
	}%
	% Set the paragraph indentation
	\setlength{\parindent}{3mm}%
	\setlength{\parskip}{0pt}%
}{%
	% Set the geometry
	\ifthenelse{\boolean{aaltoseriesvertlayout}}{%
		\geometry{%
			paper=a4paper,%
			left=5.5cm,%
			right=2.5cm,%
			top=3cm,%
			bottom=3.5cm,%
			headheight=8pt,%
			headsep=\the\aaltoseriestmplength,%
			footskip=1.2cm%
		}}{%
		\geometry{%
			paper=a4paper,%
			left=5.5cm,%
			right=2.5cm,%
			top=1.5cm,%
			bottom=2.7cm,%
			headheight=8pt,%
			headsep=\the\aaltoseriestmplength,%
		}%
	}%

	% Set the paragraph indentation
	\setlength{\parindent}{3mm}%
	\setlength{\parskip}{0mm}%
	% Set the baseline stretch
	\ifthenelse{\boolean{aaltoseriesvertlayout}}{%
		\renewcommand{\baselinestretch}{1.1}%
	}{%
		\renewcommand{\baselinestretch}{1.4}%
	}%
}

%% Set the page bottoms to be ragged
\raggedbottom

%% Set the sectioning commands
\gdef\AaltoSeriesChapterHeader{}
\newlength{\AaltoSeriesChapterHeadHeight}
\setlength{\AaltoSeriesChapterHeadHeight}{6.5cm}
\setcounter{secnumdepth}{2}
\makeatletter
\renewcommand\part{%
	\cleardoublepage%
	\thispagestyle{empty}%
	\@tempswafalse%
	\secdef\@part\@spart}
\def\@part[#1]#2{%
	\ifnum\c@secnumdepth>-2\relax%
		\refstepcounter{part}%
		\addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
	\else%
		\addcontentsline{toc}{part}{#1}%
	\fi%
	\markboth{}{}%
    {\noindent%
	\begin{minipage}{\textwidth}
		\bfseries\sffamily\fontsize{20pt}{20pt}\selectfont\flushleft\vskip1mm%
		\partname\nobreakspace\thepart%
	\end{minipage}\vskip0mm\null%
	\addtolength{\AaltoSeriesChapterHeadHeight}{-\pagetotal}%
	\addtolength{\AaltoSeriesChapterHeadHeight}{-8mm}%
	\vspace*{\AaltoSeriesChapterHeadHeight}%
	\begin{flushleft}%
		\bfseries\fontsize{20pt}{20pt}\selectfont\sffamily #2%
	\end{flushleft}%
	\vskip22pt}%
	\setlength{\AaltoSeriesChapterHeadHeight}{6.5cm}%
	\@endpart%
}
\def\@spart#1{%
	\markboth{}{}%
    {\addtolength{\AaltoSeriesChapterHeadHeight}{-12mm}%
	\vspace*{\AaltoSeriesChapterHeadHeight}%
	\begin{flushleft}%
		\bfseries\fontsize{20pt}{20pt}\selectfont\sffamily #1%
	\end{flushleft}%
	\vskip22pt}%
	\setlength{\AaltoSeriesChapterHeadHeight}{6.5cm}%
	\@endpart%
}
\renewcommand\chapter{%
	\cleardoublepage%
	\thispagestyle{plain}%
	\global\@topnum\z@%
	\@afterindentfalse%
	\secdef\@chapter\@schapter%
}
\def\@chapter[#1]#2{%
	\ifnum\c@secnumdepth>\m@ne
		\refstepcounter{chapter}%
		\addcontentsline{toc}{chapter}{\protect\numberline{\thechapter.}#1}%
	\else%
		\addcontentsline{toc}{chapter}{#1}%
	\fi%
	\addtocontents{lof}{\protect\addvspace{10\p@}}%
	\addtocontents{lot}{\protect\addvspace{10\p@}}%
	\chaptermark{#1}%
	\@makechapterhead{#2}%
	\@afterheading%
}
\def\@makechapterhead#1{%
	{\noindent%
	\setlength{\linewidth}{\textwidth}%
	\ifnum\c@secnumdepth>\m@ne%
		\begin{minipage}[t]{3em}%
			\fontsize{20pt}{20pt}\selectfont\bfseries\sffamily\flushleft\vskip-3mm%
			\thechapter.%
		\end{minipage}%
		\addtolength{\linewidth}{-3em}%
	\fi%
	\begin{minipage}[t]{\linewidth}%
		\fontsize{20pt}{20pt}\selectfont\bfseries\sffamily\flushleft\vskip-3mm #1%
	\end{minipage}\vskip0mm\null%
	\setlength{\linewidth}{\textwidth}%
	\addtolength{\AaltoSeriesChapterHeadHeight}{-\pagetotal}%
	\addtolength{\AaltoSeriesChapterHeadHeight}{-3mm}%
	\vspace*{\AaltoSeriesChapterHeadHeight}%
	\AaltoSeriesChapterHeader\gdef\AaltoSeriesChapterHeader{}}%
	\setlength{\AaltoSeriesChapterHeadHeight}{6.5cm}%
}
\def\@schapter#1{%
	\@makeschapterhead{#1}%
	\@afterheading%
}
\def\@makeschapterhead#1{%
	{\noindent%
	\begin{minipage}{\textwidth}%
		\fontsize{20pt}{20pt}\selectfont\bfseries\sffamily\flushleft\vskip-3mm #1%
	\end{minipage}\vskip0mm\null%
	\chaptermark{#1}%
	\addtolength{\AaltoSeriesChapterHeadHeight}{-\pagetotal}%
	\addtolength{\AaltoSeriesChapterHeadHeight}{-3mm}%
	\vspace*{\AaltoSeriesChapterHeadHeight}%
	\AaltoSeriesChapterHeader\gdef\AaltoSeriesChapterHeader{}}%
	\setlength{\AaltoSeriesChapterHeadHeight}{6.5cm}%
}
\renewcommand\section{%
	\@startsection{section}{1}{0mm}{-1pt}{15.4pt}{%
		\vskip29.8pt\flushleft\bfseries\sffamily\fontsize{12pt}{12pt}\selectfont}%
}
\renewcommand\subsection{%
	\@startsection{subsection}{2}{0mm}{-1pt}{15.4pt}{%
		\vskip14.4pt\flushleft\bfseries\rmfamily\fontsize{11pt}{11pt}\selectfont}%
}
\renewcommand\subsubsection{%
	\@startsection{subsubsection}{3}{0mm}{-1pt}{0.1pt}{%
		\vskip14.4pt\flushleft\itshape\rmfamily\fontsize{11pt}{11pt}\selectfont}%
}
\renewcommand\paragraph{%
	\@startsection{paragraph}{4}{0mm}{-1pt}{-1em}{%
		\flushleft\bfseries\sffamily\normalsize}%
}
\renewcommand\subparagraph{%
	\@startsection{subparagraph}{5}{0mm}{-1pt}{-1em}{%
		\flushleft\itshape\rmfamily\normalsize}%
}

% Nicer ToC, more space for numbers
\usepackage[titles]{tocloft}% 
\setlength\cftchapindent{0em}
\setlength\cftchapnumwidth{2.0em}
\setlength\cftsecindent{2.0em}
\setlength\cftsecnumwidth{3.0em}
\setlength\cftsubsecindent{5.0em}
\setlength\cftsubsecnumwidth{4.2em}
\setlength\cftfigindent{2.0em}
\setlength\cftfignumwidth{3.0em}
\setlength\cfttabindent{2.0em}
\setlength\cfttabnumwidth{3.0em}

\setlength{\cftbeforetoctitleskip}{-0.2in} % Fix too much white above Contents

\renewcommand{\cfttoctitlefont}{\fontsize{20pt}{20pt}\selectfont\bfseries\sffamily}
\renewcommand{\cftloftitlefont}{\fontsize{20pt}{20pt}\selectfont\bfseries\sffamily}
\renewcommand{\cftlottitlefont}{\fontsize{20pt}{20pt}\selectfont\bfseries\sffamily}

%%% Set the header and footer
\makeatletter
\def\ps@myheadings{%
	\def\@oddfoot{\hfil\fontsize{11pt}{11pt}\selectfont\thepage}%
	\def\@evenfoot{\fontsize{11pt}{11pt}\selectfont\thepage\hfil}%
	\def\@oddhead{\hfil\fontsize{8pt}{8pt}\selectfont\leftmark}%
	\def\@evenhead{\fontsize{8pt}{8pt}\selectfont\leftmark\hfil}%
	\let\@mkboth\@gobbletwo%
	\let\sectionmark\@gobble%
}
\def\ps@plain{%
	\def\@oddfoot{\hfil\fontsize{11pt}{11pt}\selectfont\thepage}%
	\def\@evenfoot{\fontsize{11pt}{11pt}\selectfont\thepage\hfil}%
	\let\@oddhead\@empty%
	\let\@evenhead\@empty%
	\let\@mkboth\@gobbletwo%
	\let\sectionmark\@gobble%
}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\pagestyle{myheadings}
\makeatother

%% Set the footnote command
\makeatletter
\renewcommand{\@makefntext}[1]{%
	\fontsize{10pt}{10pt}\selectfont%
	\noindent%
	{\@textsuperscript{\footnotesize\@thefnmark}}#1%
}
\makeatother

%% Set the marginpar command
\let\AaltoSeriesMarginPar\marginpar
\renewcommand{\marginpar}[2][]{%
	\ifthenelse{\equal{#1}{}}{}{\footnote{#1}\textsuperscript{, }}%
	\footnote{#2}%
}
    
%% Renew the \maketitle command
\renewcommand{\maketitle}{}

%% Renew the \author, \title, and \date commands
\gdef\AaltoSeriesAuthor{}
\gdef\AaltoSeriesSAuthor{}
\renewcommand{\author}[2][]{%
	\ifthenelse{\equal{#1}{}}{\gdef\AaltoSeriesSAuthor{#2}}{\gdef\AaltoSeriesSAuthor{#1}}%
	\gdef\AaltoSeriesAuthor{#2}%
}
\gdef\AaltoSeriesTitle{}
\gdef\AaltoSeriesSTitle{}
\renewcommand{\title}[2][]{%
	\ifthenelse{\equal{#1}{}}{\gdef\AaltoSeriesSTitle{#2}}{\gdef\AaltoSeriesSTitle{#1}}%
	\gdef\AaltoSeriesTitle{#2}%
}

%% New environment preface
\def\prefacename{Preface}
\def\prefacedefaultlocation{Helsinki}
\newcommand{\AaltoSeriesPrefaceToTOCCommand}[2]{%
	\addcontentsline{toc}{#1}{#2}%
}
\makeatletter
\AtBeginDocument{%
		\@ifpackagewith{babel}{finnish}{%
			\addto\captionsfinnish{%
				\def\prefacedefaultlocation{Helsinki}%
			% This fixes a bug in finnish.ldf (missing percent sign)
			\def\abstractname{Tiivistelm\"a}%
			}%
		}{}%
		\@ifpackagewith{babel}{swedish}{%
			\addto\captionsswedish{%
				\def\prefacedefaultlocation{Helsingfors}%
			}%
		}{}%
		\@ifpackagewith{babel}{english}{%
			\addto\captionsenglish{%
				\def\prefacedefaultlocation{Helsinki}%
			}%
		}{}%
}
\newenvironment{preface}[2][\prefacedefaultlocation]{%
	\gdef\AaltoSeriesPrefaceAuthor{\AaltoSeriesAuthor~#2}
	\ifthenelse{\boolean{aaltoseriesnochapters} \and \not\boolean{aaltoseriesarticlecollection}}{%
		\section*{\prefacename}%
		\phantomsection%
		\AaltoSeriesPrefaceToTOCCommand{section}{\prefacename}%
	}{%
		\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
		\chapter*{\prefacename}%
		\phantomsection%
		\AaltoSeriesPrefaceToTOCCommand{chapter}{\prefacename}%
		\vspace*{6.5cm}%
	}%
	\gdef\AaltoSeriesPrefaceEnding{#1}%
}{%
	\vskip2\baselineskip%
	\noindent\ifthenelse{\equal{\AaltoSeriesPrefaceEnding}{}}{}{\AaltoSeriesPrefaceEnding, }\@date,\vskip2\baselineskip%
	\noindent\AaltoSeriesPrefaceAuthor%
	\gdef\AaltoSeriesPrefaceEnding{}
}
\makeatother

%% Renew the abstract environment
\ifthenelse{\boolean{aaltoseriesabstractsonly}}{%
	\renewenvironment{abstract}[1][]{%
		\clearpage%
		\fontsize{9pt}{10.8pt}\selectfont%
		\ifthenelse{\equal{#1}{}}{}{\selectlanguage{#1}}%
		\noindent\ignorespaces%
	}{%
		\normalfont\noindent%
	}%
}{%
	\renewenvironment{abstract}[1][]{%
		\vspace{\baselineskip}%
		{\noindent\bfseries\fontsize{12pt}{12pt}\selectfont\abstractname}%
		\vskip\baselineskip%
		\ifthenelse{\equal{#1}{}}{}{\selectlanguage{#1}}%
		\itshape\noindent\ignorespaces%
	}{%
		\normalfont\vspace{\baselineskip}\noindent%
	}%
}

%% New \draftabstract command for drafting
\newcommand{\draftabstract}[2][]{}

%% Renew the verse, quote, and quotation environments
\renewenvironment{verse}{%
	\begin{list}{}{\topsep\baselineskip\rightmargin0mm\leftmargin3mm\itemindent0mm\listparindent3mm\parsep0mm}%
	\item\relax
	\fontsize{10pt}{14pt}\selectfont\itshape%
}{%
	\end{list}%
	\ignorespacesafterend%
}
\renewenvironment{quote}{%
	\begin{list}{}{\topsep\baselineskip\rightmargin0mm\leftmargin3mm\itemindent0mm\listparindent0mm\parsep0mm}%
	\item\relax
	\fontsize{10pt}{14pt}\selectfont%
}{%
	\end{list}%
	\ignorespacesafterend%
}
\renewenvironment{quotation}{%
	\begin{list}{}{\topsep\baselineskip\rightmargin0mm\leftmargin3mm\itemindent3mm\listparindent3mm\parsep0mm}%
	\item\relax
	\fontsize{10pt}{14pt}\selectfont%
	\ignorespaces%
}{%
	\end{list}%
	\ignorespacesafterend%
}

%% Renew the itemize, enumerate, and description environments
\newlength{\AaltoSeriesListSep}\setlength{\AaltoSeriesListSep}{\baselineskip}
\newcounter{AaltoSeriesItemizeScope}\setcounter{AaltoSeriesItemizeScope}{0}
\newcounter{AaltoSeriesEnumerateScope}\setcounter{AaltoSeriesEnumerateScope}{0}
\renewenvironment{itemize}{%
	\stepcounter{AaltoSeriesItemizeScope}%
	\ifthenelse{\arabic{AaltoSeriesItemizeScope} = 1}{%
		\begin{list}{\labelitemi}{%
			\topsep\baselineskip\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
		}%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
	\ifthenelse{\arabic{AaltoSeriesItemizeScope} = 2}{%
		\begin{list}{\labelitemii}{%
			\topsep0mm\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
		}%
		\vskip\AaltoSeriesListSep%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
	\ifthenelse{\arabic{AaltoSeriesItemizeScope} = 3}{%
		\begin{list}{\labelitemiii}{%
			\topsep0mm\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
		}%
		\vskip\AaltoSeriesListSep%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
	\ifthenelse{\arabic{AaltoSeriesItemizeScope} > 3}{%
		\begin{list}{\labelitemiv}{%
			\topsep0mm\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
		}%
		\vskip\AaltoSeriesListSep%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
}{%
	\setlength{\AaltoSeriesListSep}{\itemsep}%
	\end{list}%
	\addtocounter{AaltoSeriesItemizeScope}{-1}%
	\noindent\ignorespacesafterend%
}
\renewenvironment{enumerate}{%
	\stepcounter{AaltoSeriesEnumerateScope}%
	\ifthenelse{\arabic{AaltoSeriesEnumerateScope} = 1}{%
		\begin{list}{\labelenumi}{%
			\topsep\baselineskip\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
			\usecounter{enumi}%
		}%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
	\ifthenelse{\arabic{AaltoSeriesEnumerateScope} = 2}{%
		\begin{list}{\labelenumii}{%
			\topsep0mm\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
			\usecounter{enumii}%
		}%
		\vskip\AaltoSeriesListSep%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
	\ifthenelse{\arabic{AaltoSeriesEnumerateScope} = 3}{%
		\begin{list}{\labelenumiii}{%
			\topsep0mm\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
			\usecounter{enumiii}%
		}%
		\vskip\AaltoSeriesListSep%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
	\ifthenelse{\arabic{AaltoSeriesEnumerateScope} = 4}{%
		\begin{list}{\labelenumiv}{%
			\topsep0mm\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
			\usecounter{enumiv}%
		}%
		\vskip\AaltoSeriesListSep%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
	\ifthenelse{\arabic{AaltoSeriesEnumerateScope} > 4}{%
		\begin{list}{}{%
			\topsep0mm\partopsep0mm%
			\leftmargin3mm\rightmargin0mm%
			\labelwidth1.5mm\labelsep1.5mm%
			\parsep0mm\listparindent3mm%
		}%
		\vskip\AaltoSeriesListSep%
		\setlength{\itemsep}{\baselineskip}%
	}{}%
}{%
	\setlength{\AaltoSeriesListSep}{\itemsep}%
	\end{list}%
	\addtocounter{AaltoSeriesEnumerateScope}{-1}%
	\noindent\ignorespacesafterend%
}
\newenvironment{singleitems}{%
	\begin{itemize}%
	\setlength{\itemsep}{0mm}%
}{%
	\end{itemize}%
}
\newenvironment{singleenums}{%
	\begin{enumerate}%
	\setlength{\itemsep}{0mm}%
}{%
	\end{enumerate}%
}

%% Set the caption formatting
\RequirePackage{caption}
\DeclareCaptionFont{AaltoSeriesCaption}{\fontsize{9pt}{9pt}\selectfont}
\captionsetup{format=hang,labelformat=default,labelsep=period,textformat=simple,%
	justification=justified,singlelinecheck=false,font={AaltoSeriesCaption},labelfont={bf}}
	
%% Lengths for the overwide figures, tables, and general floats
\newlength{\widefigurewidth}\setlength{\widefigurewidth}{\textwidth}
\newlength{\widetablewidth}\setlength{\widetablewidth}{\textwidth}
\newlength{\widefloatwidth}\setlength{\widefloatwidth}{\textwidth}

%% Command for the shift
\newlength{\overwidefloatshift}
\newlength{\AaltoSeriesTMPLength}\setlength{\AaltoSeriesTMPLength}{-3cm}
\newcounter{aaltoserieswidefloat}\setcounter{aaltoserieswidefloat}{1}
\newcommand{\widefloatshift}[1][1]{%
	\setlength{\widefloatwidth}{\textwidth}%
	\setlength{\overwidefloatshift}{#1\AaltoSeriesTMPLength}%
	\label{aaltoserieswidefloat\arabic{aaltoserieswidefloat}}%
	\ifthenelse{\isodd{\pageref{aaltoserieswidefloat\arabic{aaltoserieswidefloat}}}}{%
		\ifthenelse{\lengthtest{\overwidefloatshift < \AaltoSeriesTMPLength}}{%
			\setlength{\overwidefloatshift}{\AaltoSeriesTMPLength}%
			\hspace*{\overwidefloatshift}%
			\addtolength{\widefloatwidth}{-\overwidefloatshift}%
		}{%
			\setlength{\AaltoSeriesTMPLength}{0cm}%
			\ifthenelse{\lengthtest{\overwidefloatshift > \AaltoSeriesTMPLength}}{%
				\setlength{\overwidefloatshift}{0cm}%
			}{%
				\hspace*{\overwidefloatshift}%
				\addtolength{\widefloatwidth}{-\overwidefloatshift}%
			}%
		}%
	}{%
		\ifthenelse{\lengthtest{\overwidefloatshift < \AaltoSeriesTMPLength}}{%
			\setlength{\overwidefloatshift}{\AaltoSeriesTMPLength}%
			\addtolength{\widefloatwidth}{-\overwidefloatshift}%
		}{%
			\setlength{\AaltoSeriesTMPLength}{0cm}%
			\ifthenelse{\lengthtest{\overwidefloatshift > \AaltoSeriesTMPLength}}{%
				\setlength{\overwidefloatshift}{0cm}%
			}{%
				\addtolength{\widefloatwidth}{-\overwidefloatshift}%
			}%
		}%
	}%
	\setlength{\AaltoSeriesTMPLength}{-3cm}%
	\stepcounter{aaltoserieswidefloat}%
	\ignorespaces%
}
\newcommand{\widefigureshift}[1][1]{%
	\widefloatshift[#1]%
	\setlength{\widefigurewidth}{\widefloatwidth}%
}
\newcommand{\widetableshift}[1][1]{%
	\widefloatshift[#1]%
	\setlength{\widetablewidth}{\widefloatwidth}%
}

%% Renew the bibliography environment
\makeatletter
\renewcommand{\bibname}{References}
\renewenvironment{thebibliography}[1]{%
	\ifthenelse{\boolean{aaltoseriesnochapters}}{%
		\section*{\bibname}%
		\phantomsection%
	}{%
		\phantomsection%
		\chapter*{\bibname}%
		\chaptermark{\bibname}%
	}%
	\begin{list}{\@biblabel{\@arabic\c@enumiv}}{%
		\settowidth\labelwidth{\@biblabel{#1}}%
		\leftmargin\labelwidth%
		\advance\leftmargin\labelsep%
		\@openbib@code%
		\usecounter{enumiv}%
		\let\p@enumiv\@empty%
		\renewcommand\theenumiv{\@arabic\c@enumiv}}
		\fontsize{10pt}{10pt}\selectfont%
}{%
	\end{list}%
}%
\makeatother
\AtBeginDocument{%
	\let\AaltoSeriesBibliography\bibliography%
	\ifthenelse{\isundefined{\bibsection}}{}{%
		\ifthenelse{\boolean{aaltoseriesnochapters}}{%
			\renewcommand{\bibsection}{%
				\section*{\bibname}%
			}%
		}{%
			\renewcommand{\bibsection}{%
				\chapter*{\bibname}%
				\chaptermark{\bibname}%
			}%
		}%
	}%
	\renewcommand{\bibliography}[1]{%
		{\fontsize{10pt}{10pt}\selectfont%
		\cleardoublepage
		\phantomsection%
		\AaltoSeriesBibliography{#1}}%
	}%
}

%% Include the table of contents automatically to the table of contents
\newcommand{\AaltoSeriesTOCToTOCCommand}[2]{%
	\phantomsection%
	\addcontentsline{toc}{#1}{#2}%
}
\makeatletter
\renewcommand{\tableofcontents}{%
	\ifthenelse{\boolean{aaltoseriesnochapters}\and\not\boolean{aaltoseriesarticlecollection}}{%
		\section*{\contentsname}%
	}{%
		\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
		\chapter*{\contentsname}%
		\AaltoSeriesTOCToTOCCommand{chapter}{\contentsname}%
		\chaptermark{\contentsname}%
		\vspace*{6.5cm}%
	}%
	\@starttoc{toc}%
	\vskip22pt%
}
\renewcommand{\listoftables}{%
	\ifthenelse{\boolean{aaltoseriesnochapters}\and\not\boolean{aaltoseriesarticlecollection}}{%
		\section*{\listtablename}%
	}{%
		\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
		\chapter*{\listtablename}%
		\AaltoSeriesTOCToTOCCommand{chapter}{\listtablename}%
		\chaptermark{\listtablename}%
		\vspace*{6.5cm}%
	}%
	\@starttoc{lot}%
	\vskip22pt%
}
\renewcommand{\listoffigures}{%
	\ifthenelse{\boolean{aaltoseriesnochapters}\and\not\boolean{aaltoseriesarticlecollection}}{%
		\section*{\listfigurename}%
	}{%
		\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
		\chapter*{\listfigurename}%
		\AaltoSeriesTOCToTOCCommand{chapter}{\listfigurename}%
		\vspace*{6.5cm}%
	}%
	\@starttoc{lof}%
	\vskip22pt%
}
\makeatother

\def\abbrevname{Abbreviations}%
\newcommand{\abbreviations}{%
	\ifthenelse{\boolean{aaltoseriesnochapters}\and\not\boolean{aaltoseriesarticlecollection}}{%
		\section*{\abbrevname}%
	}{%
		\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
		\chapter*{\abbrevname}%
		\AaltoSeriesTOCToTOCCommand{chapter}{\abbrevname}%
		\chaptermark{\abbrevname}%
		\vspace*{6.5cm}%
	}%
}
\def\symbolsname{Symbols}%
\newcommand{\symbols}{%
	\ifthenelse{\boolean{aaltoseriesnochapters}\and\not\boolean{aaltoseriesarticlecollection}}{%
		\section*{\symbolsname}%
	}{%
		\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
		\chapter*{\symbolsname}%
		\AaltoSeriesTOCToTOCCommand{chapter}{\symbolsname}%
		\chaptermark{\symbolsname}%
		\vspace*{6.5cm}%
	}%
}


%% Include everything else to the table of contents automatically
\ifthenelse{\boolean{aaltoseriesnochapters}}{%
	\RequirePackage[nottoc,notlof,notlot,notbib]{tocbibind}%
}{%
	\RequirePackage[nottoc,notlof,notlot]{tocbibind}%
}

%% Define a command to check whether packages are loaded
\makeatletter
\let\IfAaltoSeriesPackageLoaded\@ifpackageloaded 
\makeatother

%% Execute mode-dependent command definitions and redefinitions
% Article collection specific commands and redefinitions
\makeatletter
\ifthenelse{\boolean{aaltoseriesarticlecollection}}{%
	% Packages for including the pdf articles
	\RequirePackage{ifpdf}
	\ifpdf%
		\RequirePackage{pdfpages}
		\includepdfset{noautoscale=true,fitpaper}
	\fi
	% Set the table of contents depth to the article title level
	\setcounter{tocdepth}{0}
	% Renew the commands that are used to typeset the header of specific topics to the TOC
	\renewcommand{\AaltoSeriesPrefaceToTOCCommand}[2]{%
		\addtocontents{toc}{%
			\string\noindent\string\bfseries\string\makebox[2.3em][l]{\thepage}%
			\string\sffamily\relax\prefacename\string\newline%
			\string\mdseries\string\rmfamily%
			}%
	}%
	\renewcommand{\AaltoSeriesTOCToTOCCommand}[2]{%
		\phantomsection%
		\addtocontents{toc}{%
			\string\noindent\string\bfseries\string\makebox[2.3em][l]{\thepage}%
			\string\sffamily\relax\contentsname\string\newline%
			\string\mdseries\string\rmfamily%
			}%
	}%
	% New commands for the author affiliations
	\gdef\AaltoSeriesAffiliation{}%
	\newcommand{\aff}[1]{\textsuperscript{\normalfont #1}}
	\newcommand{\affiliation}[1]{\gdef\AaltoSeriesAffiliation{#1}}
	% Reference commands
	\newcommand{\refart}[2][]{%
		\ifthenelse{\equal{#1}{}}{\ref{#2-title}}{\ref{#2-stitle}, #1}%
	}%
	\newcommand{\ra}[2][]{%
		\ifthenelse{\equal{#1}{}}{\ref{#2-stitle}}{\ref{#2-stitle}, #1}%
	}%
	\newcommand{\refartauth}[1]{%
		\ifthenelse{\equal{\ref{#1-author}}{\ref{#1-sauthor}}}{\ref{#1-author}}{\ref{#1-sauthor}}%
	}%
	\newcommand{\raa}[1]{\ref{#1-sauthor}}%
	% Command for the PDF bookmark
	\newcommand{\articlepdfbookmark}[1]{}%
	% Command for the label
	\newcommand{\articlelabels}[1]{%
		\string\newlabel{#1}{{\AaltoSeriesTitle}{\thepage}}%
		\string\newlabel{#1-title}{{\AaltoSeriesTitle}{\thepage}}%
		\string\newlabel{#1-stitle}{{\AaltoSeriesSTitle}{\thepage}}%
		\string\newlabel{#1-author}{{\AaltoSeriesAuthor}{\thepage}}%
		\string\newlabel{#1-sauthor}{{\AaltoSeriesSAuthor}{\thepage}}%
	}%
	% Renew the maketitle command to typeset the article title and author information
	\renewcommand{\maketitle}[1][]{%
		\gdef\AaltoSeriesChapterHeader{%
			\vskip0pt%
			\sffamily\fontsize{11pt}{11pt}\selectfont%
			\bfseries\sffamily\noindent\AaltoSeriesAuthor%
			\vskip22pt%
			\mdseries\noindent\AaltoSeriesAffiliation%
			\normalfont%
			\vskip22pt%
		}%
		\stepcounter{chapter}%
		\chapter*{\AaltoSeriesTitle}%
		\ifthenelse{\equal{#1}{}}{}{%
			\articlepdfbookmark{#1}%
			\protected@write\@auxout{}{%
				\articlelabels{#1}%
			}%
		}%
		\setlength{\linewidth}{\textwidth}\addtolength{\linewidth}{-2.3em}%
		\addtocontents{toc}{%
			\string\vskip11pt%
			\string\noindent\string\parbox[t]{2.3em}{\string\bfseries \thepage}%
			\string\parbox[t]{\string\linewidth}{\string\bfseries\string\sffamily%
			\string\hspace{0pt}\AaltoSeriesTitle\string\newline%
			\string\mdseries\string\rmfamily\string\itshape%
			\string\hspace{0pt}\AaltoSeriesSAuthor}%
		}%
		\setlength{\linewidth}{\textwidth}%
		\chaptermark{\AaltoSeriesSAuthor: \AaltoSeriesSTitle}%
	}%
	% Define command \addpdfarticle for articles in pdf format
	\newcommand{\addpdfarticle}[7][]{%
		\ifthenelse{\isodd{\thepage}}{\cleardoublepage}{\clearpage}%
		\stepcounter{chapter}%
		\def\AaltoSeriesTitle{#4}%
		\def\AaltoSeriesAuthor{#5}%
		\addtocontents{toc}{%
			\string\vskip11pt%
			\string\noindent\string\bfseries\string\makebox[2.3em][l]{\string\bfseries \thepage}%
			\string\sffamily\string\hspace{0pt}\AaltoSeriesTitle\string\newline%
			\string\mdseries\string\rmfamily\string\itshape\string\hspace*{2.3em}\AaltoSeriesAuthor%
			\string\upshape%
		}%
		\ifthenelse{\equal{#6}{}}{\def\AaltoSeriesSTitle{#4}}{\def\AaltoSeriesSTitle{#6}}%
		\ifthenelse{\equal{#7}{}}{\def\AaltoSeriesSAuthor{#5}}{\def\AaltoSeriesSAuthor{#7}}%
		\ifpdf%
			\ifthenelse{\equal{#3}{}}{}{%
				\articlepdfbookmark{#3}%
				\protected@write\@auxout{}{%
					\articlelabels{#3}%
				}%
			}%
			\chaptermark{}%
			\includepdf[pagecommand={},#1]{#2}%
			\chaptermark{\AaltoSeriesSAuthor: \AaltoSeriesSTitle}%
			\includepdf[pages=2-,pagecommand={},#1]{#2}%
		\else%
			\ClassWarning{aaltoseries}{The PDF document is not included to the document}%
			\null\vfill%
			{\Huge\bfseries USE pdf\LaTeX\ TO INCLUDE THE PDF DOCUMENT!}
			\vfill\null%
		\fi%
	}%
	\AtBeginDocument{%
		% Renew couple of commands if hyperref is loaded
		\IfAaltoSeriesPackageLoaded{hyperref}{%
			% Command for the PDF bookmark
			\renewcommand{\articlepdfbookmark}[1]{%
				\pdfbookmark[1]{\AaltoSeriesTitle}{art.#1}%
			}%
			% Command for the label
			\renewcommand{\articlelabels}[1]{%
				\string\newlabel{#1}{{\AaltoSeriesTitle}{\thepage}{\AaltoSeriesTitle}{art.#1.1}{}}%
				\string\newlabel{#1-title}{{\AaltoSeriesTitle}{\thepage}{\AaltoSeriesTitle}{art.#1.1}{}}%
				\string\newlabel{#1-stitle}{{\AaltoSeriesSTitle}{\thepage}{\AaltoSeriesSTitle}{art.#1.1}{}}%
				\string\newlabel{#1-author}{{\AaltoSeriesAuthor}{\thepage}{\AaltoSeriesAuthor}{art.#1.1}{}}%
				\string\newlabel{#1-sauthor}{{\AaltoSeriesSAuthor}{\thepage}{\AaltoSeriesSAuthor}{art.#1.1}{}}%
			}%
		}{}%
	}%
}{}
\makeatletter

% PhD thesis specific commands and redefinitions
\makeatletter
\ifthenelse{\boolean{aaltoseriesdissertation}}{%
	% Package for including the pdf articles
	\RequirePackage{ifpdf}
	\ifpdf%
		\RequirePackage{pdfpages}
		\includepdfset{noautoscale=true,fitpaper}
	\fi
	% Package for the uppercasing the first letter
	\RequirePackage{mfirstuc}
	\newcommand{\Uppercasefirst}{\xmakefirstuc}
	% Commands to cite the publication
	\newcommand{\citepub}[2][]{\ref{#2-long}\ifthenelse{\equal{#1}{}}{}{, #1}}%
	\newcommand{\cpub}[2][]{\ref{#2-short}\ifthenelse{\equal{#1}{}}{}{, #1}}%
	\newcommand{\cp}[2][]{\ref{#2-number}\ifthenelse{\equal{#1}{}}{}{, #1}}%
	% Counter for the publications
	\newcounter{aaltoseriesdissertationpublication}\setcounter{aaltoseriesdissertationpublication}{0}%
	% Formatting variables for the publication
	\gdef\AaltoSeriesDissertationPublicationPunctuationPostPreBook{}%
	\gdef\AaltoSeriesDissertationPublicationPunctuationFirst{.}%
	\gdef\AaltoSeriesDissertationPublicationPunctuationSecond{.}%
	\gdef\AaltoSeriesDissertationPublicationPunctuationThird{,}%
	\gdef\AaltoSeriesDissertationPublicationPunctuationFourth{,}%
	\gdef\AaltoSeriesDissertationPublicationPunctuationFifth{}%
	\gdef\AaltoSeriesDissertationPublicationPunctuationSixth{.}% Added BH
	\gdef\AaltoSeriesDissertationPublicationFormatPreBook{\upshape\Uppercasefirst}%
	\gdef\AaltoSeriesDissertationPublicationFormatFirst{}%
	\gdef\AaltoSeriesDissertationPublicationFormatSecond{}%
	\gdef\AaltoSeriesDissertationPublicationFormatThird{\em}%
	\gdef\AaltoSeriesDissertationPublicationFormatFourth{}%
	\gdef\AaltoSeriesDissertationPublicationFormatFifth{}%
	\gdef\AaltoSeriesDissertationPublicationFormatSixth{}%
	\gdef\AaltoSeriesDissertationPublicationPreFirst{}%
	\gdef\AaltoSeriesDissertationPublicationPreSecond{}%
	\gdef\AaltoSeriesDissertationPublicationPreThird{}%
	\gdef\AaltoSeriesDissertationPublicationPreFourth{}%
	\gdef\AaltoSeriesDissertationPublicationPreFifth{}%
	\gdef\AaltoSeriesDissertationPublicationPreSixth{}%
	\gdef\AaltoSeriesDissertationPublicationPostFirst{}%
	\gdef\AaltoSeriesDissertationPublicationPostSecond{}%
	\gdef\AaltoSeriesDissertationPublicationPostThird{}%
	\gdef\AaltoSeriesDissertationPublicationPostFourth{}%
	\gdef\AaltoSeriesDissertationPublicationPostFifth{}%
	\gdef\AaltoSeriesDissertationPublicationPostSixth{}%
	\gdef\AaltoSeriesDissertationPublicationAuthor{1}%
	\gdef\AaltoSeriesDissertationPublicationTitle{2}%
	\gdef\AaltoSeriesDissertationPublicationPublication{3}%
	\gdef\AaltoSeriesDissertationPublicationPublicationInfo{4}%
	\gdef\AaltoSeriesDissertationPublicationMonth{5}%
	\gdef\AaltoSeriesDissertationPublicationYear{6}%
	% Command to redefine the publication punctuation
	\newcommand{\publicationpunctuation}[6][]{%
		\gdef\AaltoSeriesDissertationPublicationPunctuationPostPreBook{#1}%
		\gdef\AaltoSeriesDissertationPublicationPunctuationFirst{#2}%
		\gdef\AaltoSeriesDissertationPublicationPunctuationSecond{#3}%		
		\gdef\AaltoSeriesDissertationPublicationPunctuationThird{#4}%
		\gdef\AaltoSeriesDissertationPublicationPunctuationFourth{#5}%
		\gdef\AaltoSeriesDissertationPublicationPunctuationFifth{#6}%		
	}%
	% Command to redefine the publication formatting
	\newcommand{\publicationformatting}[7][]{%
		\gdef\AaltoSeriesDissertationPublicationFormatPreBook{#1}%
		\gdef\AaltoSeriesDissertationPublicationFormatFirst{#2}%
		\gdef\AaltoSeriesDissertationPublicationFormatSecond{#3}%
		\gdef\AaltoSeriesDissertationPublicationFormatThird{#4}%
		\gdef\AaltoSeriesDissertationPublicationFormatFourth{#5}%
		\gdef\AaltoSeriesDissertationPublicationFormatFifth{#6}%
		\gdef\AaltoSeriesDissertationPublicationFormatSixth{#7}%
	}%
	% Command to redefine the preformatting of the publication information
	\newcommand{\publicationadditionalpreformatting}[6]{%
		\gdef\AaltoSeriesDissertationPublicationPreFirst{#1}%
		\gdef\AaltoSeriesDissertationPublicationPreSecond{#2}%
		\gdef\AaltoSeriesDissertationPublicationPreThird{#3}%
		\gdef\AaltoSeriesDissertationPublicationPreFourth{#4}%
		\gdef\AaltoSeriesDissertationPublicationPreFifth{#5}%
		\gdef\AaltoSeriesDissertationPublicationPreSixth{#6}%
	}%
	% Command to redefine the postformatting of the publicaiton information
	\newcommand{\publicationadditionalpostformatting}[6]{%
		\gdef\AaltoSeriesDissertationPublicationPostFirst{#1}%
		\gdef\AaltoSeriesDissertationPublicationPostSecond{#2}%
		\gdef\AaltoSeriesDissertationPublicationPostThird{#3}%
		\gdef\AaltoSeriesDissertationPublicationPostFourth{#4}%
		\gdef\AaltoSeriesDissertationPublicationPostFifth{#5}%
		\gdef\AaltoSeriesDissertationPublicationPostSixth{#6}%
	}%
	% Commands to order the publication data
	\newcommand{\publicationdataselector}[2]{%
		\ifthenelse{\equal{#1}{author}}{\gdef\AaltoSeriesDissertationPublicationAuthor{#2}}{}%
		\ifthenelse{\equal{#1}{title}}{\gdef\AaltoSeriesDissertationPublicationTitle{#2}}{}%
		\ifthenelse{\equal{#1}{publication}}{\gdef\AaltoSeriesDissertationPublicationPublication{#2}}{}%
		\ifthenelse{\equal{#1}{information}}{\gdef\AaltoSeriesDissertationPublicationPublicationInfo{#2}}{}%
		\ifthenelse{\equal{#1}{month}}{\gdef\AaltoSeriesDissertationPublicationMonth{#2}}{}%
		\ifthenelse{\equal{#1}{year}}{\gdef\AaltoSeriesDissertationPublicationYear{#2}}{}%
	}%
	\newcommand{\publicationdataorder}[6]{%
		\publicationdataselector{#1}{1}%
		\publicationdataselector{#2}{2}%
		\publicationdataselector{#3}{3}%
		\publicationdataselector{#4}{4}%
		\publicationdataselector{#5}{5}%
		\publicationdataselector{#6}{6}%
	}%
	\newcommand{\publicationgetpreformatting}[1]{%
		\ifthenelse{#1 = 1}{\AaltoSeriesDissertationPublicationPreFirst}{}%
		\ifthenelse{#1 = 2}{\AaltoSeriesDissertationPublicationPreSecond}{}%
		\ifthenelse{#1 = 3}{\AaltoSeriesDissertationPublicationPreThird}{}%
		\ifthenelse{#1 = 4}{\AaltoSeriesDissertationPublicationPreFourth}{}%
		\ifthenelse{#1 = 5}{\AaltoSeriesDissertationPublicationPreFifth}{}%
		\ifthenelse{#1 = 6}{\AaltoSeriesDissertationPublicationPreSixth}{}%
	}%
	\newcommand{\publicationgetformatting}[1]{%
		\ifthenelse{#1 = 1}{\AaltoSeriesDissertationPublicationFormatFirst}{}%
		\ifthenelse{#1 = 2}{\AaltoSeriesDissertationPublicationFormatSecond}{}%
		\ifthenelse{#1 = 3}{\AaltoSeriesDissertationPublicationFormatThird}{}%
		\ifthenelse{#1 = 4}{\AaltoSeriesDissertationPublicationFormatFourth}{}%
		\ifthenelse{#1 = 5}{\AaltoSeriesDissertationPublicationFormatFifth}{}%
		\ifthenelse{#1 = 6}{\AaltoSeriesDissertationPublicationFormatSixth}{}%
	}%
	\newcommand{\publicationgetpostformatting}[1]{%
		\ifthenelse{#1 = 1}{\AaltoSeriesDissertationPublicationPostFirst}{}%
		\ifthenelse{#1 = 2}{\AaltoSeriesDissertationPublicationPostSecond}{}%
		\ifthenelse{#1 = 3}{\AaltoSeriesDissertationPublicationPostThird}{}%
		\ifthenelse{#1 = 4}{\AaltoSeriesDissertationPublicationPostFourth}{}%
		\ifthenelse{#1 = 5}{\AaltoSeriesDissertationPublicationPostFifth}{}%
		\ifthenelse{#1 = 6}{\AaltoSeriesDissertationPublicationPostSixth}{}%
	}%
	\newcommand{\publicationgetpunctuation}[1]{%
		\ifthenelse{#1 = 1}{\AaltoSeriesDissertationPublicationPunctuationFirst\space}{}%
		\ifthenelse{#1 = 2}{\AaltoSeriesDissertationPublicationPunctuationSecond\space}{}%
		\ifthenelse{#1 = 3}{\AaltoSeriesDissertationPublicationPunctuationThird\space}{}%
		\ifthenelse{#1 = 4}{\AaltoSeriesDissertationPublicationPunctuationFourth\space}{}%
		\ifthenelse{#1 = 5}{\AaltoSeriesDissertationPublicationPunctuationFifth\space}{}%
		\ifthenelse{#1 = 6}{.}{}%
	}%
	\newcommand{\publicationgetdata}[8][]{%
		\ifthenelse{\equal{#2}{\AaltoSeriesDissertationPublicationAuthor}}{%
			\publicationgetpreformatting{#2}{\publicationgetformatting{#2}#3}%
			\publicationgetpostformatting{#2}\publicationgetpunctuation{#2}%
		}{}%
		\ifthenelse{\equal{#2}{\AaltoSeriesDissertationPublicationTitle}}{%
			\publicationgetpreformatting{#2}{\publicationgetformatting{#2}#4}%
			\publicationgetpostformatting{#2}\publicationgetpunctuation{#2}%
		}{}%
		\ifthenelse{\equal{#2}{\AaltoSeriesDissertationPublicationPublication}}{%
			\ifthenelse{\equal{#1}{}}{}{%
				{\AaltoSeriesDissertationPublicationFormatPreBook%
				#1\AaltoSeriesDissertationPublicationPunctuationPostPreBook}\space%
			}%
			\publicationgetpreformatting{#2}{\publicationgetformatting{#2}#5}%
			\publicationgetpostformatting{#2}\publicationgetpunctuation{#2}%
		}{}%
		\ifthenelse{\equal{#2}{\AaltoSeriesDissertationPublicationPublicationInfo}}{%
			\ifthenelse{\equal{#1}{\AaltoSeriesDissertationSubmittedPrefix}}{}{%
				\ifthenelse{\equal{#6}{}}{}{%
					\publicationgetpreformatting{#2}{\publicationgetformatting{#2}#6}%
					\publicationgetpostformatting{#2}\publicationgetpunctuation{#2}%
				}%
			}%
		}{}%
		\ifthenelse{\equal{#2}{\AaltoSeriesDissertationPublicationMonth}}{%
			\publicationgetpreformatting{#2}{\publicationgetformatting{#2}#7}%
			\publicationgetpostformatting{#2}\publicationgetpunctuation{#2}%
		}{}%
		\ifthenelse{\equal{#2}{\AaltoSeriesDissertationPublicationYear}}{%
			\publicationgetpreformatting{#2}{\publicationgetformatting{#2}#8}%
			\publicationgetpostformatting{#2}\publicationgetpunctuation{#2}%
		}{}%
	}%
	% Command to format the publication, used internally by some other commands
	\newcommand{\publication}[7][]{%
		\publicationgetdata[#1]{1}{#2}{#3}{#4}{#5}{#6}{#7}%
		\publicationgetdata[#1]{2}{#2}{#3}{#4}{#5}{#6}{#7}%
		\publicationgetdata[#1]{3}{#2}{#3}{#4}{#5}{#6}{#7}%
		\publicationgetdata[#1]{4}{#2}{#3}{#4}{#5}{#6}{#7}%
		\publicationgetdata[#1]{5}{#2}{#3}{#4}{#5}{#6}{#7}%
		\publicationgetdata[#1]{6}{#2}{#3}{#4}{#5}{#6}{#7}%
	}%
	% Label names for the publications
	\def\publicationname{Publication}
	\def\publicationsname{Publications}
	\def\pubname{P}
	% Label name for conference articles
	\def\AaltoSeriesDissertationConferencePrefix{in}
	% Label name for accepted publications
	\def\AaltoSeriesDissertationAcceptedPrefix{accepted for publication in}
	% Label name for submitted publications
	\def\AaltoSeriesDissertationSubmittedPrefix{submitted to}
	% Label for the reprint permission
	\def\reprintname{Reprinted with permission}
	% Formatting variables for the Author's Contribution list
	\gdef\AaltoSeriesContributionPreFormatting{``}%
	\gdef\AaltoSeriesContributionPostFormatting{''}%
	% Command for changing the formatting variables in Author's Contribution list
	\newcommand{\contributionformatting}[2]{%
		\gdef\AaltoSeriesContributionPreFormatting{#1}%
		\gdef\AaltoSeriesContributionPostFormatting{#2}%
	}%
	% Command for the PDF bookmark
	\newcommand{\publicationpdfbookmark}[1]{}%
	% Command for the label
	\newcommand{\publicationlabels}[1]{%
		\string\newlabel{#1}{{\publicationname\ \Roman{aaltoseriesdissertationpublication}}{\thepage}}%
		\string\newlabel{#1-long}{{\publicationname\ \Roman{aaltoseriesdissertationpublication}}{\thepage}}%
		\string\newlabel{#1-short}{{\pubname\Roman{aaltoseriesdissertationpublication}}{\thepage}}%
		\string\newlabel{#1-number}{{\Roman{aaltoseriesdissertationpublication}}{\thepage}}%
	}%
	% Command for adding a publication
	\newcommand{\addpublication}[9][]{%
		\stepcounter{aaltoseriesdissertationpublication}%
		\ifthenelse{\equal{#1}{conference}}{%
			\addtocontents{asp}{\string\item\space\string\publication[\AaltoSeriesDissertationConferencePrefix]{#2}{#3}{#4}{#5}{#6}{#7}}%
		}{%
			\ifthenelse{\equal{#1}{accepted}}{%
				\addtocontents{asp}{\string\item\space\string\publication[\AaltoSeriesDissertationAcceptedPrefix]{#2}{#3}{#4}{#5}{#6}{#7}}%
			}{%
%				\ifthenelse{\equal{#1}{submitted}\and\boolean{aaltoseriesdraft}}{%
				\ifthenelse{\equal{#1}{submitted}}{%
					\addtocontents{asp}{\string\item\space\string\publication[\AaltoSeriesDissertationSubmittedPrefix]{#2}{#3}{#4}{#5}{#6}{#7}}%
				}{%
					\addtocontents{asp}{\string\item\space\string\publication{#2}{#3}{#4}{#5}{#6}{#7}}%
				}%
			}%
		}%
		\ifthenelse{\boolean{aaltoseriesdissertationnocontribution}}{}{%
			\immediate\@writefile{asc}{{\gdef\AaltoSeriesContributionTitle{\AaltoSeriesContributionPreFormatting #3\AaltoSeriesContributionPostFormatting}}}%
			\immediate\write\tf@asc{\string\section*{\publicationname\ \Roman{aaltoseriesdissertationpublication}: {\string\AaltoSeriesContributionTitle}}}%
		}%
		\gdef\AaltoSeriesChapterHeader{%
			\vskip0pt\AaltoSeriesMarginPar{\vspace*{-6.6cm}\rule{2.5cm}{6.5cm}}%
%			{\bfseries\sffamily\fontsize{11pt}{11pt}\selectfont%
			{\mdseries\sffamily\fontsize{12pt}{12pt}\selectfont% PH
			\ifthenelse{\equal{#1}{conference}}{%
				\noindent\publication[\AaltoSeriesDissertationConferencePrefix]{#2}{#3}{#4}{#5}{#6}{#7}%
			}{%
				\ifthenelse{\equal{#1}{accepted}}{%
					\noindent\publication[\AaltoSeriesDissertationAcceptedPrefix]{#2}{#3}{#4}{#5}{#6}{#7}%
				}{%
					\ifthenelse{\equal{#1}{submitted}}{%
						\noindent\publication[\AaltoSeriesDissertationSubmittedPrefix]{#2}{#3}{#4}{#5}{#6}{#7}%
					}{%
						\noindent\publication{#2}{#3}{#4}{#5}{#6}{#7}%
					}%
				}%
			}}%
		}%
		\cleardoublepage%
		\phantomsection%
		\chapter*{\publicationname\  \Roman{aaltoseriesdissertationpublication}}%
		\ifthenelse{\arabic{aaltoseriesdissertationpublication} = 1}{%
			\addcontentsline{toc}{chapter}{\publicationsname}%
		}{}%
		\vskip-8.5cm%
		\publicationpdfbookmark{#9}%
		\vskip8.5cm%
		\vskip22pt%
		\ifthenelse{\equal{#1}{submitted}}{}{%
			{\sffamily\fontsize{11pt}{11pt}\selectfont%
%			\textcopyright\ #7 #8.\\\reprintname.}\normalfont%
			\textcopyright\ #7 #8\\\reprintname.}\normalfont% Dot removed (PH) as it can be added manually
		}%
		\pagestyle{empty}%
		\ifthenelse{\boolean{aaltoseriesdissertationnopagenumbers}}{%
			\thispagestyle{empty}%
		}{%
			\thispagestyle{plain}%
		}%
		\protected@write\@auxout{}{%
			\publicationlabels{#9}%
		}%
		%%\publicationpdfbookmark{#9}%
	}%
	% Command for including the pdf publication
	\newcommand{\addpublicationpdf}[2][]{%
		\ifpdf%
			\includepdf[openright,pages=-,#1]{#2}%
		\else
			\ClassWarning{aaltoseries}{The required PDF document not included to the document}%
			\null\vfill%
			{\Huge\bfseries USE pdf\LaTeX\ TO INCLUDE THE PDF DOCUMENT!}
			\vfill\null%
		\fi%
	}%
	% Command for including a pdf supplement of a publication
	\newcommand{\addsupplementpdf}[2][]{%
		\ifpdf%
			\includepdf[pages=-,#1]{#2}%
		\else
			\ClassWarning{aaltoseries}{The required PDF document not included to the document}%
			\null\vfill%
			{\Huge\bfseries USE pdf\LaTeX\ TO INCLUDE THE PDF DOCUMENT!}
			\vfill\null%
		\fi%
	}%
	% Command for adding the authors contribution
	\newcommand{\addcontribution}[1]{%
		\@writefile{asc}{{#1}}%
	}%
	% Label for the list of publications
	\def\listofpublicationsname{List of Publications}%
	% Text that is in the beginning of the list of publications
	\def\listofpublicationspreamble{%
		This thesis consists of an overview and of the following publications which are referred to in the text by their Roman numerals.%
	}%
	% Label for the author's contribution
	\def\authorscontributionname{Author's Contribution\\ \color{red}\large Please check the guidelines on author's contribution from your own doctoral programme}%
	% Person or company responsible for language checking
	\newcommand{\languagecheck}[1]{%
		\def\languagechecker{#1}%
	}%
	% Command for typesetting the list of publication and author's contribution (if not suppressed)
	\newcommand{\listofpublications}{%
		\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
		\chapter*{\listofpublicationsname}%
		\phantomsection%
		\addcontentsline{toc}{chapter}{\listofpublicationsname}%
		\vspace*{6.5cm}%
		\listofpublicationspreamble%
		\begin{enumerate}%
		\renewcommand{\labelenumi}{\bfseries\Roman{enumi}}%
		\newread\aspread\immediate\openin\aspread=\jobname.asp%
		\ifeof\aspread\item[]%
		\else\fi\closein\aspread%
		\@starttoc{asp}%
		\end{enumerate}%
		\ifthenelse{\boolean{aaltoseriesdissertationnocontribution}}{}{%
			\setlength{\AaltoSeriesChapterHeadHeight}{0cm}%
			\chapter*{\authorscontributionname}%
			\phantomsection%
			\addcontentsline{toc}{chapter}{\authorscontributionname}%
			\vspace*{6.5cm}%
			\@starttoc{asc}%
			\ifthenelse{\equal{\languagechecker}{no}}{}{%
				\languagecheckack			
			}%
		}%
	}%
	% Label for errata list
	\def\errataname{Errata}%
	% Command for adding errata for a publication
	\newcommand{\adderrata}[1]{%
		\immediate\write\tf@ase{{\string\section*{\publicationname\ \Roman{aaltoseriesdissertationpublication}}}}%
		\@writefile{ase}{{#1}}%
	}%
	% Command for typesetting the errata list
	\newcommand{\errata}[1][]{%
		\cleardoublepage%
		\phantomsection%
		\chapter*{\vspace*{4mm}\errataname}%
		\addcontentsline{toc}{chapter}{\errataname}%
		\ifthenelse{\equal{#1}{}}{}{#1}%
		\@starttoc{ase}%
	}%
	% Additional modifications to be performed at the beginning of the document
	\AtBeginDocument{%
		\@ifpackagewith{babel}{finnish}{%
			\addto\captionsfinnish{%
				\def\abbrevname{Lyhenteet}%
				\def\symbolsname{Symbolit}%
				\def\listofpublicationsname{Julkaisuluettelo}%
				\def\listofpublicationspreamble{%
					T{\"a}m{\"a} v{\"a}it{\"o}skirja koostuu yhteenvedosta ja seuraavista julkaisuista, joihin viitataan tekstiss{\"a} roomalaisin numeroin.%
				}%
				\def\publicationname{Julkaisu}%
				\def\publicationsname{Julkaisut}%
				\def\pubname{J}%
				\def\AaltoSeriesDissertationConferencePrefix{kirjassa}%
				\def\AaltoSeriesDissertationAcceptedPrefix{hyv{\"a}ksytty julkaistavaksi julkaisussa}%
				\def\AaltoSeriesDissertationSubmittedPrefix{l{\"a}hetetty julkaisuun}%
				\def\authorscontributionname{Tekij{\"a}n osuus\\ \color{red}\large Please check the guidelines on author's contribution from your own doctoral programme}%
				\def\reprintname{Julkaistu luvalla}%
				\def\languagecheckack{%
					\section*{Kielentarkastus}
					Työni on oikolukenut/kielentarkastanut \languagechecker. %
					Olen itse hyväksynyt/hylännyt kielentarkastajan ehdottamat muutokset yksitellen. %
					Työni tieteellinen sisältö ei ole tästä muuttunut.%
				}%
			}%
		}{}%
		\@ifpackagewith{babel}{swedish}{%
			\addto\captionsswedish{%
				\def\abbrevname{F\"orkortningar}%
				\def\symbolsname{Symboler}%
				\def\listofpublicationsname{Publikationsf{\"o}rteckning}%
				\def\listofpublicationspreamble{%
					Denna doktorsavhandling best{\aa}r av en {\"o}versikt och i f{\"o}ljande publikationer som avses i texten som deras romerska siffror.%
				}%
				\def\publicationname{Publikation}%
				\def\publicationsname{Publikationer}%
				\def\pubname{P}%
				\def\AaltoSeriesDissertationConferencePrefix{i}%
				\def\AaltoSeriesDissertationAcceptedPrefix{accepterad f{\"o}r publicering i}%
				\def\AaltoSeriesDissertationSubmittedPrefix{l{\"a}mnats till}%
				\def\authorscontributionname{F{\"o}rfattarens bidrag\\ \color{red}\large Please check the guidelines on author's contribution from your own doctoral programme}%
				\def\reprintname{Publiceras med tillst{\aa}nd}%
				\def\languagecheckack{%
					\section*{Språkgranskning}
					Min avhandling har språkgranskats av \languagechecker. %
					Jag har personligen granskat och accepterat/förkastat resultaterna av språkgransningen ett för ett. %
					Detta har inte ändrat vetenskapliga innehållet av min avhandling.%
				}%
			}%
		}{}%
		\@ifpackagewith{babel}{english}{%
			\addto\captionsenglish{%
				\def\abbrevname{Abbreviations}%
				\def\symbolsname{Symbols}%
				\def\listofpublicationsname{List of Publications}%
				\def\listofpublicationspreamble{%
					This thesis consists of an overview and of the following publications which are referred to in the text by their Roman numerals.%
				}%
				\def\publicationname{Publication}%
				\def\publicationsname{Publications}%
				\def\pubname{P}%
				\def\AaltoSeriesDissertationConferencePrefix{in}%
				\def\AaltoSeriesDissertationAcceptedPrefix{accepted for publication in}%
				\def\AaltoSeriesDissertationSubmittedPrefix{submitted to}%
				\def\authorscontributionname{Author's Contribution\\ \color{red}\large Please check the guidelines on author's contribution from your own doctoral programme}%
				\def\reprintname{Reprinted with permission}%
				\def\languagecheckack{%
					\section*{Language check}
					The language of my dissertation has been checked by \languagechecker. %
					I have personally examined and accepted/rejected the results of the language check one by one. %
					This has not affected the scientific content of my dissertation.%
				}%
			}%
		}{}%
		% Renew couple of commands if hyperref is loaded
		\IfAaltoSeriesPackageLoaded{hyperref}{%
			\hypersetup{bookmarksopen,bookmarksnumbered,colorlinks,plainpages=false} % !plainpages means inter-page links work
			% Command for the PDF bookmark
			\renewcommand{\publicationpdfbookmark}[1]{%
				\pdfbookmark[1]{\publicationname\ \Roman{aaltoseriesdissertationpublication}}{pub.#1}%
			}%
			% Command for the label
			\renewcommand{\publicationlabels}[1]{%
				\string\newlabel{#1}{{\publicationname\ \Roman{aaltoseriesdissertationpublication}}{\thepage}{\publicationname\ \Roman{aaltoseriesdissertationpublication}}{pub.#1.1}{}}%
				\string\newlabel{#1-long}{{\publicationname\ \Roman{aaltoseriesdissertationpublication}}{\thepage}{\publicationname\ \Roman{aaltoseriesdissertationpublication}}{pub.#1.1}{}}%
				\string\newlabel{#1-short}{{\pubname\Roman{aaltoseriesdissertationpublication}}{\thepage}{\pubname\ \Roman{aaltoseriesdissertationpublication}}{pub.#1.1}{}}%
				\string\newlabel{#1-number}{{\Roman{aaltoseriesdissertationpublication}}{\thepage}{\Roman{aaltoseriesdissertationpublication}}{pub.#1.1}{}}%
			}%
		}{}%
	}%
}{}
\makeatother

% Report specific commands and redefinitions
\ifthenelse{\boolean{aaltoseriesreport}}{%
	% Command to specify the author of the given chapter
	\newcommand{\chapterauthor}[1]{%
		\gdef\AaltoSeriesChapterHeader{\vskip0pt\mdseries\rmfamily\fontsize{12pt}{12pt}\selectfont\noindent #1\normalfont\vskip22pt}%
	}%
}{}

% Teaching material specific commands and redefinitions
\ifthenelse{\boolean{aaltoseriesteachingmaterial}}{%
	% None, proposals welcome
}{}

% Working paper specific commands and redefinitions
\ifthenelse{\boolean{aaltoseriesworkingpaper}}{%
	% None, proposals welcome
}{}

% Individual article specific commands and redefinitions (articles for an article collection)
\ifthenelse{\boolean{aaltoseriesarticle}}{%
	% New commands for the author affiliations
	\gdef\AaltoSeriesAffiliation{}%
	\newcommand{\aff}[1]{\textsuperscript{\normalfont #1}}%
	\newcommand{\affiliation}[1]{\gdef\AaltoSeriesAffiliation{#1}}%
	% Renew the maketitle command to typeset the article title and author information
	\renewcommand{\maketitle}{%
		\gdef\AaltoSeriesChapterHeader{%
			\vskip0pt%
			\sffamily\fontsize{11pt}{11pt}\selectfont%
			\bfseries\sffamily\noindent\AaltoSeriesAuthor%
			\vskip22pt%
			\mdseries\noindent\AaltoSeriesAffiliation%
			\normalfont%
			\vskip22pt%
		}%
		\chapter*{\AaltoSeriesTitle}%
		\chaptermark{\AaltoSeriesSAuthor: \AaltoSeriesSTitle}%
		\thispagestyle{empty}%
	}%
	% Remove the page numbering
	\AtBeginDocument{%
		\pagestyle{empty}%
	}%
}{}

% Short publication specific commands and redefinitions
\ifthenelse{\boolean{aaltoseriesnochapters}}{%
	% Don't use chapter numbers in sections
	\renewcommand{\thesection}{\arabic{section}}%
	% Don't use chapter numbers in figure and table captions
	\renewcommand{\thefigure}{\arabic{figure}}%
	\renewcommand{\thetable}{\arabic{table}}%
	% Don't use chapter numbers in equation numbers
	\renewcommand{\theequation}{\arabic{equation}}%
}{%
	% Use chapter numbers in figure and table captions
	\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}%
	\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}%
	% Use chapter numbers in equation numbers
	\renewcommand{\theequation}{\arabic{chapter}.\arabic{equation}}%
}

% Draft specific commands and redefinitions
\makeatletter
\ifthenelse{\boolean{aaltoseriesdraft}}{%
	% Renew the abstract command
	\renewcommand{\draftabstract}[2][]{%
		\clearpage%
		\thispagestyle{empty}%
		{\noindent\fontsize{12pt}{12pt}\selectfont\bfseries\ifthenelse{\equal{#1}{}}{}{\selectlanguage{#1}}\abstractname}
		\vskip11pt%
		{\noindent\fontsize{10pt}{10pt}\selectfont\ifthenelse{\equal{#1}{}}{}{\selectlanguage{#1}}\noindent#2}%
		\vskip22pt%
	}%
	\def\draftname{DRAFT}
	\AtBeginDocument{%
	% Add the publication information to the front page
		\null\vskip5.5cm%
		\thispagestyle{empty}%
		{\fontsize{14pt}{14pt}\selectfont\bfseries\noindent (DRAFT)}%
		\vskip44pt%
		{\fontsize{20pt}{20pt}\selectfont\bfseries\sffamily\noindent\AaltoSeriesTitle}%
		\vskip22pt%
		{\fontsize{12pt}{12pt}\selectfont\itshape\noindent\AaltoSeriesAuthor}%
		\vskip22pt%
		{\fontsize{11pt}{11pt}\selectfont\noindent\today}
		\clearpage%
	}%
}{}
\makeatother

%% Check whether the babel package is used
\AtBeginDocument{%
	\IfAaltoSeriesPackageLoaded{babel}{}{%
		% Set the command
		\renewcommand{\selectlanguage}[1]{}%
	}%
}
