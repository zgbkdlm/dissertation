\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{zz}[2020/10/11 Zheng Zhao's minimalism beamer class]

% Dependencies
\RequirePackage{lastpage}
\RequirePackage{calc}
\RequirePackage[dvipsnames]{xcolor}

% Commands and definitions
\newlength\titlepagesep \setlength\titlepagesep{0.2cm}
\newlength\titlepageauthorsep \setlength\titlepageauthorsep{0.6cm}
\newlength\footery \setlength\footery{8cm}

\definecolor{footergray}{gray}{0.5}
\newcommand{\setcmapBeijing}{%
	\definecolor{frametitlecolor}{gray}{0.2}
	\definecolor{mastercolour}{named}{RoyalPurple}
	\definecolor{secondcolour}{RGB}{255, 121, 19}
}
\newcommand{\setcmapHelsinki}{%
	\definecolor{frametitlecolor}{gray}{0.2}
	\definecolor{mastercolour}{RGB}{66, 140, 212}
	\definecolor{secondcolour}{RGB}{255, 156, 218}
}
\newcommand{\setcmapReykjavik}{%
	\definecolor{frametitlecolor}{gray}{0.2}
	\definecolor{mastercolour}{gray}{0.4}
	\definecolor{secondcolour}{gray}{0.4}
}
\setcmapHelsinki

\newif\ifseriffont\seriffontfalse
\newif\iffullfooter\fullfooterfalse

% Parse options and load beamer
\DeclareOption{garamond}{%
	}
\DeclareOption{seriffont}{\seriffonttrue}
\DeclareOption{fullfooter}{\fullfootertrue}
\DeclareOption{cmap=Beijing}{\setcmapBeijing}
\DeclareOption{cmap=Helsinki}{\setcmapHelsinki}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax
\LoadClass{beamer}

% Commands and definitions that depend on options
\renewcommand{\titlepage}{%
	{%
		\setbeamertemplate{footline}{}
		\frame[t, noframenumbering]{%
			\vspace{2cm}
			\centering{
				{\Large \scshape \textbf{\inserttitle}}\\[0.4cm]
				\insertsubtitle\\[1.8cm]
				\insertauthor\\[\titlepageauthorsep]
				{\scriptsize \insertinstitute}\\[\titlepagesep]
				{\scriptsize \insertdate}
			}
		}
	}
}

% Beamer customisations
\iffullfooter
\newcommand{\footertext}{\beamer@shorttitle}
\else
\newcommand{\footertext}{~}
\fi
\setbeamertemplate{footline}{%
	\noindent
	\begin{minipage}{.45\paperwidth}
		\vspace{-0.5cm}
		\hspace{\beamer@leftmargin}
		\footertext
	\end{minipage}
	\hfill
	\begin{minipage}{.45\paperwidth}
		\vspace{-0.5cm}
		\hspace{.35\paperwidth minus \beamer@rightmargin}
		{%
			\color{footergray}
			\tiny
			\arabic{page}/\pageref{LastPage}
		}
	\end{minipage}
}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{frametitle}{%
	{%
		\color{frametitlecolor}
		\vspace{0.2cm}\insertframetitle\\[-0.15cm]
		\rule{\widthof{\insertframetitle}}{1.5pt}
	}
}

% Fonts
\ifseriffont
%	\usefonttheme{structuresmallcapsserif}
	\usefonttheme{serif}
\fi
\setbeamerfont{section title}{size=\normalsize,series=\bfseries}
\setbeamerfont{frametitle}{series=\bfseries, shape=\scshape, family=\rmfamily}
%\setbeamerfont{framesubtitle}{series=\rmfamily}
%\setbeamerfont{caption}{series=\rmfamily}
%\AtBeginDocument{\rmfamily}

% Colours
\setbeamercolor{structure}{fg=mastercolour}
\setbeamercolor{alerted text}{fg=secondcolour}
\setbeamercolor{example text}{fg=mastercolour}
