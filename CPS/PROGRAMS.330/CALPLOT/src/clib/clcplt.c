/*
c---------------------------------------------------------------------c
c                                                                     c
c      COMPUTER PROGRAMS IN SEISMOLOGY                                c
c      VOLUME I                                                       c
c                                                                     c
c      PROGRAM: CLCPLT                                                c
c                                                                     c
c      COPYRIGHT (C)  1997 R. B. Herrmann                             c
c                                                                     c
c      Department of Earth and Atmospheric Sciences                   c
c      Saint Louis University                                         c
c      3507 Laclede Avenue                                            c
c      St. Louis, Missouri 63103                                      c
c      U. S. A.                                                       c
c                                                                     c
c---------------------------------------------------------------------c
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "dosubs.h"
#include "disubs.h"
FILE *grfp;	/* this must be seen outside */
int is_long;
#include "dpsubs.h"
#ifdef MSDOS
#include <io.h>
#include <fcntl.h>
#define GETPID rand
#else
#define GETPID getpid
#endif
#include "calplot.h"

/* signal handling */
#include <signal.h>
#ifdef	MSDOS
	static int sigs[]= { SIGINT, SIGTERM };
#else
	static int sigs[]= { SIGHUP, SIGINT , SIGQUIT, SIGBUS, SIGKILL,
		SIGABRT, SIGTERM, SIGSEGV, SIGFPE, SIGILL };
#endif
void onintr(int arg);
static int sig_set = 0;

/* get function definitions */
/* set up pointers to graphics functions 
	that do the work */

void	(*do_arc)(INT Xi,INT Yi,INT X0,INT Y0,INT X1,INT Y1)=0,
	(*do_circle)(INT Xi,INT Yi,INT r)=0,
	(*do_clip)(INT cmd, INT X0,INT Y0,INT X1,INT Y1)=0,
	(*do_closepl)( int mode)=0,
	(*do_cont)(INT X0,INT Y0)=0,
	(*do_control)(int type, int i1, int i2, int i3, int i4)=0,
	(*do_cross)(int *X0,int *Y0, char *c)=0,
        (*do_cursor)(INT curstyp)=0,
	(*do_erase)(INT mode)=0,
	(*do_fillp)(INT n,INT *x,INT *y)=0,
	(*do_fillr)(INT X0,INT Y0,INT X1,INT Y1,
		INT patx,INT paty,INT lenx,INT leny)=0,
	(*do_fills)(INT X0,INT Y0,INT ixy,INT istnd,INT iplmn)=0,
	(*do_fillt)(INT X0,INT Y0,INT X1,INT Y1,INT X2,INT Y2,
		INT patx,INT paty,INT lenx,INT leny)=0,
	(*do_font)(INT Xi)=0,
	(*do_gottxt)(char *s)=0,
	(*do_info)(int *HasMouse, INT *XminDev, INT *YminDev, 
		INT *XmaxDev, INT *YmaxDev, INT *XminClip, 
		INT *YminClip, INT *XmaxClip, INT *YmaxClip, INT *Color)=0,
	(*do_gsymb)(INT X0,INT Y0,INT X1,INT Y1,INT n,char *s)=0,
	(*do_gwid)(INT wid)=0,
	(*do_label)(char *s)=0,
	(*do_linec)(INT X0,INT Y0,INT X1,INT Y1)=0,
	(*do_linemod)(char *s)=0,
	(*do_move)(INT X0,INT Y0)=0,
	(*do_openpl)( int mode)=0,
	(*do_pen)(INT Xi)=0,
	(*do_point)(INT X0,INT Y0)=0,
	(*do_space)(INT X0,INT Y0,INT X1,INT Y1)=0;

#define YES 1
#define NO  0


static int plotfile = 0;
/* prototypes */
INT dp_ClipRightSave, dp_ClipLeftSave, dp_ClipTopSave, dp_ClipBottomSave;
static void dgclip(INT icmd, INT ilw, INT jlw, INT iup, INT jup);
static void order(INT *x,INT *y);

void clospl(int mode)
{
	if(plotfile == 1){
        	fflush(grfp);
		fclose(grfp);
		plotfile = 0;
	} else {
		(*do_erase)(mode);
	}
		
	
}

#define NAMLEN  80
void openpl(int ls,char *s, int lis, char *iconstr)
{
        char name[NAMLEN];
	int i;
	/* set up signals for termination gracefully */
	if(sig_set == 0){
		for ( i=0; i< sizeof(sigs)/sizeof(int) ; ++i)
			if(signal( sigs[i],SIG_IGN) != SIG_IGN )
				signal ( sigs[i], onintr);
		sig_set = 1;
	}
	
	/* set up global for clipping */
	dp_ClipRightSave	=  100000000;
	dp_ClipLeftSave		= -100000000;
	dp_ClipTopSave		=  100000000;
	dp_ClipBottomSave	= -100000000;
	/* parse the name of the file string */
/*
	if(strncmp(s,"INTE",4) == 0 ){
		plotfile = 0;
		(*do_openpl)(1);
	} else {
INTERACTIVE PROCESSING
*/
		
		if(strncmp(s,"plot",4) == 0 ){	
#ifdef MSDOS
			int stime;
			long ltime;
			ltime = time(NULL);
			stime = (unsigned int)ltime/2;
			srand(stime);
			stime = rand()%9999;
			srand(stime);
        		sprintf(name,"plot%d",rand()%9999 );
#else
        		sprintf(name,"plot%d",GETPID() );
#endif
#ifdef MSDOS
        		grfp = fopen(name,"wb");
#else
        		grfp = fopen(name,"w");
#endif
			}
		else if(strncmp(s,"stdout",6) == 0){
			grfp = stdout;
#ifdef MSDOS
			setmode(fileno(stdout),O_BINARY);
#endif
			}
		else {
			/* strip off blanks at the end of the string s */
			for(i=0 ; i< ls && *s != ' ' && *s != '\0' ; i++)
				name[i] = *s++ ;
			name[i] = '\0' ;
#ifdef MSDOS
        		grfp = fopen(name,"wb");
#else
        		grfp = fopen(name,"w");
#endif
		}
		plotfile = 1;
/*
	}
*/



	/* set up the function definitions 
		for an interactive/hardcopy
		this is where the switch would occur */
	if(plotfile == 1){
		do_arc 		= dp_arc ;
		do_circle 	= dp_circle ;
		do_clip 	= dgclip ;
		do_cont 	= dp_cont ;
		do_control 	= dp_control ;
		do_cross 	= dp_cross ;
		do_erase 	= dp_erase ;
		do_fillp 	= dp_fillp ;
		do_fillr 	= dp_fillr ;
		do_fills 	= dp_fills ;
		do_fillt 	= dp_fillt ;
		do_font 	= dp_font ;
		do_gottxt 	= dp_gottxt ;
		do_cursor 	= dp_cursor ;
		do_info 	= dp_info ;
		do_gsymb 	= dp_gsymb ;
		do_gwid 	= dp_gwid ;
		do_label 	= dp_label ;
		do_linec 	= dp_linec ;
		do_linemod 	= dp_linemod ;
		do_move 	= dp_move ;
		do_pen	 	= dp_pen ;
		do_point 	= dp_point ;
		do_space 	= dp_space ;
/*
	} else {
		do_arc 		= di_arc ;
		do_circle 	= di_circle ;
		do_clip 	= dgclip ;
		do_closepl 	= di_closepl ;
		do_control 	= di_control ;
		do_cross 	= di_cross ;
		do_erase 	= di_erase ;
		do_fillp 	= di_fillp ;
		do_fillr 	= di_fillr ;
		do_fills 	= di_fills ;
		do_fillt 	= di_fillt ;
		do_font 	= di_font ;
		do_gottxt 	= di_gottxt ;
		do_cursor 	= di_cursor ;
		do_info 	= di_info ;
		do_gsymb 	= di_gsymb ;
		do_gwid 	= di_gwid ;
		do_label 	= di_label ;
		do_linec 	= di_linec ;
		do_linemod 	= di_linemod ;
		do_move 	= di_move ;
		do_openpl 	= di_openpl ;
		do_pen	 	= di_pen ;
		do_point 	= di_point ;
		do_space 	= di_space ;
*/
	}
}

#define NAMSTR  81
static char name[NAMSTR];
/* output text string to graphics terminal at current position */
void gottxt(int cnt,char *s)
{
	int j;
	j = cnt;
	if(j >= NAMSTR)j=NAMSTR-1;
	strncpy(name,s,j);
	name[j] = '\0';
	(*do_gottxt)(name);
}

/* get text string from terminal */
void gintxt(int cnt,char *s)
{
	s[0]='\0';
}

void gomesg(int cnt,char *s)
{
}


static void dgclip(INT icmd, INT ilw, INT jlw, INT iup, INT jup)
{
	if(icmd == 1){
		order(&ilw, &iup);
		order(&jlw, &jup);
	}
	dp_ClipRightSave	= iup;
	dp_ClipLeftSave		= ilw;
	dp_ClipTopSave		= jup;
	dp_ClipBottomSave	= jlw;
	if(plotfile == YES)
		dp_clip (icmd, ilw, jlw, iup, jup);
	/*
	else
		di_clip (icmd, ilw, jlw, iup, jup);
	*/
	if(icmd == 0){
		dp_ClipRightSave	=  100000000;
		dp_ClipLeftSave		= -100000000;
		dp_ClipTopSave		=  100000000;
		dp_ClipBottomSave	= -100000000;
	}
}
static void order(INT *x,INT *y)
{
	INT tmp;
	if(*y < *x){
		tmp = *x;
		*x = *y;
		*y = tmp;
	}
}

