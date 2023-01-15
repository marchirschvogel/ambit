set grid
# set grid linewidth 0.05

# size of font
#set size 0.5, 0.5

# gnuplot line colors:
# https://i.stack.imgur.com/x6yLm.png
# gnuplot dash types:
# http://gnuplot.sourceforge.net/demo_svg_5.2/dashtypes.html

# lines - start with left atrium and go through the circulation...
set style line 1 dashtype 3 linecolor rgb '#f03232' linewidth 4 # light-red - at_l
set style line 2 dashtype 1 linecolor rgb '#ff0000' linewidth 3 # red - v_l
set style line 3 dashtype 6 linecolor rgb '#8b0000' linewidth 4 # dark-red - ar_sys

set style line 4 dashtype 5 linecolor rgb '#ff4500' linewidth 4 # orange-red - arperi_sys

set style line 5 dashtype 4 linecolor rgb '#fa8072' linewidth 4 # salmon - arspl_sys
set style line 6 dashtype 2 linecolor rgb '#ffa07a' linewidth 4 # light-salmon - arespl_sys
set style line 7 dashtype 7 linecolor rgb '#e9967a' linewidth 4 # dark-salmon - armsc_sys
set style line 8 dashtype 8 linecolor rgb '#f08080' linewidth 4 # light-coral - arcer_sys
set style line 9 dashtype 9 linecolor rgb '#ff7f50' linewidth 4 # coral - arcor_sys

set style line 10 dashtype 11 linecolor rgb '#add8e6' linewidth 4 # light-blue - venspl_sys
set style line 11 dashtype 12 linecolor rgb '#87ceeb' linewidth 4 # sky-blue - venespl_sys
set style line 12 dashtype 13 linecolor rgb '#000080' linewidth 4 # navy - venmsc_sys
set style line 13 dashtype 14 linecolor rgb '#00ffff' linewidth 4 # cyan - vencer_sys
set style line 14 dashtype 15 linecolor rgb '#008b8b' linewidth 4 # dark-cyan - vencor_sys

set style line 15 dashtype 10 linecolor rgb '#4169e1' linewidth 3 # royal-blue - ven_sys
set style line 151 dashtype 11 linecolor rgb '#0000cd' linewidth 3 # medium-blue - ven2_sys

set style line 16 dashtype 3 linecolor rgb '#00ced1' linewidth 4 # dark-turquoise - at_r
set style line 17 dashtype 1 linecolor rgb '#0000ff' linewidth 3 # blue - v_r
set style line 18 dashtype 6 linecolor rgb '#191970' linewidth 4 # midnight-blue - ar_pul

set style line 19 dashtype 5 linecolor rgb '#9400d3' linewidth 3 # dark-violet - cap_pul
set style line 20 dashtype 10 linecolor rgb '#8b008b' linewidth 3 # dark-magenta - ven_pul
set style line 201 dashtype 10 linecolor rgb '#f055f0' linewidth 3 # light-magenta - ven2_pul
set style line 202 dashtype 11 linecolor rgb '#ff00ff' linewidth 3 # magenta - ven3_pul
set style line 203 dashtype 12 linecolor rgb '#ff1493' linewidth 3 # dark-pink - ven4_pul
set style line 204 dashtype 13 linecolor rgb '#804080' linewidth 3 # orchid4 - ven5_pul

set style line 21 dashtype 1 linecolor rgb '#006400' linewidth 4 # dark-green
set style line 22 dashtype 1 linecolor rgb '#a020f0' linewidth 4 # purple

set style line 23 dashtype 1 linecolor rgb '#ee82ee' linewidth 4 # violet
set style line 24 dashtype 1 linecolor rgb '#7fffd4' linewidth 4 # aquamarine

set style line 97 dashtype 2 linecolor rgb 'black' linewidth 4
set style line 98 dashtype 3 linecolor rgb 'black' linewidth 4

set style line 99 dashtype 1 linecolor rgb 'black' linewidth 4

set style line 102 dashtype (25,3,1,3) linecolor rgb '#ff0000' linewidth 4 # red - v_l
set style line 117 dashtype 3 linecolor rgb '#0000ff' linewidth 4 # blue - v_r

set style line 101 dashtype 5 linecolor rgb '#ff0000' linewidth 4 # light-red - at_l
set style line 116 dashtype 6 linecolor rgb '#0000ff' linewidth 4 # dark-turquoise - at_r

set style line 300 dashtype 1 linecolor rgb 'black' linewidth 4
set style line 301 dashtype 3 linecolor rgb 'black' linewidth 4

#set terminal unknown

set key samplen __SAMPLEN__ width __SAMPWID__ font '2,2' right top spacing 3.5 maxcolumns __MAXCOLS__ maxrows __MAXROWS__

#set ytics nomirror
#__HAVEY2__set y2tics nomirror offset -0.5

set samples 1000.

set xrange [__X1S__:__X1E__]
#__HAVEX2__set x2range [__X2S__:__X2E__]

set yrange [__Y1S__:__Y1E__]
#__HAVEY2__set y2range [__Y2S__:__Y2E__]

set terminal epslatex standalone dashlength 0.5
set output '__OUTDIR__/__OUTNAME__.tex'

set xlabel '$__X1VALUE__\;[\mathrm{__X1UNIT__}]$' offset 0,0.25
set ylabel '$__Y1VALUE__\;[\mathrm{__Y1UNIT__}]$' offset 0.5,0
#__HAVEX2__set x2label '$__X2VALUE__\;[\mathrm{__X2UNIT__}]$' offset -1.0,0
#__HAVEY2__set y2label '$__Y2VALUE__\;[\mathrm{__Y2UNIT__}]$' offset -1.0,0

FILENAME01 = '__FILEDIR__/__QTY1__.txt'
TITLE01    = '__TIT1__'
LINE01     = __LIN1__

#__2__FILENAME02 = '__FILEDIR__/__QTY2__.txt'
#__2__TITLE02    = '__TIT2__'
#__2__LINE02     = __LIN2__

#__3__FILENAME03 = '__FILEDIR__/__QTY3__.txt'
#__3__TITLE03    = '__TIT3__'
#__3__LINE03     = __LIN3__

#__4__FILENAME04 = '__FILEDIR__/__QTY4__.txt'
#__4__TITLE04    = '__TIT4__'
#__4__LINE04     = __LIN4__

#__5__FILENAME05 = '__FILEDIR__/__QTY5__.txt'
#__5__TITLE05    = '__TIT5__'
#__5__LINE05     = __LIN5__

#__6__FILENAME06 = '__FILEDIR__/__QTY6__.txt'
#__6__TITLE06    = '__TIT6__'
#__6__LINE06     = __LIN6__

#__7__FILENAME07 = '__FILEDIR__/__QTY7__.txt'
#__7__TITLE07    = '__TIT7__'
#__7__LINE07     = __LIN7__

#__8__FILENAME08 = '__FILEDIR__/__QTY8__.txt'
#__8__TITLE08    = '__TIT8__'
#__8__LINE08     = __LIN8__

#__9__FILENAME09 = '__FILEDIR__/__QTY9__.txt'
#__9__TITLE09    = '__TIT9__'
#__9__LINE09     = __LIN9__

#__10__FILENAME10 = '__FILEDIR__/__QTY10__.txt'
#__10__TITLE10    = '__TIT10__'
#__10__LINE10     = __LIN10__

#__11__FILENAME11 = '__FILEDIR__/__QTY11__.txt'
#__11__TITLE11    = '__TIT11__'
#__11__LINE11     = __LIN11__

#__12__FILENAME12 = '__FILEDIR__/__QTY12__.txt'
#__12__TITLE12    = '__TIT12__'
#__12__LINE12     = __LIN12__

#__13__FILENAME13 = '__FILEDIR__/__QTY13__.txt'
#__13__TITLE13    = '__TIT13__'
#__13__LINE13     = __LIN13__

#__14__FILENAME14 = '__FILEDIR__/__QTY14__.txt'
#__14__TITLE14    = '__TIT14__'
#__14__LINE14     = __LIN14__

#__15__FILENAME15 = '__FILEDIR__/__QTY15__.txt'
#__15__TITLE15    = '__TIT15__'
#__15__LINE15     = __LIN15__

#__16__FILENAME16 = '__FILEDIR__/__QTY16__.txt'
#__16__TITLE16    = '__TIT16__'
#__16__LINE16     = __LIN16__

#__17__FILENAME17 = '__FILEDIR__/__QTY17__.txt'
#__17__TITLE17    = '__TIT17__'
#__17__LINE17     = __LIN17__

#__18__FILENAME18 = '__FILEDIR__/__QTY18__.txt'
#__18__TITLE18    = '__TIT18__'
#__18__LINE18     = __LIN18__

### 18 different graphs in one plot should be enough to consider...

plot FILENAME01 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE01 with lines ls LINE01#__2__, FILENAME02 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE02 with lines ls LINE02#__3__, FILENAME03 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE03 with lines ls LINE03#__4__, FILENAME04 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE04 with lines ls LINE04#__5__, FILENAME05 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE05 with lines ls LINE05#__6__, FILENAME06 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE06 with lines ls LINE06#__7__, FILENAME07 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE07 with lines ls LINE07#__8__, FILENAME08 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE08 with lines ls LINE08#__9__, FILENAME09 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE09 with lines ls LINE09#__10__, FILENAME10 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE10 with lines ls LINE10#__11__, FILENAME11 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE11 with lines ls LINE11#__12__, FILENAME12 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE12 with lines ls LINE12#__13__, FILENAME13 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE13 with lines ls LINE13#__14__, FILENAME14 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE14 with lines ls LINE14#__15__, FILENAME15 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE15 with lines ls LINE15#__16__, FILENAME16 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE16 with lines ls LINE16#__17__, FILENAME17 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE17 with lines ls LINE17#__18__, FILENAME18 every 1 using (__XSCALE__*$1):(__YSCALE__*$2) title TITLE18 with lines ls LINE18
