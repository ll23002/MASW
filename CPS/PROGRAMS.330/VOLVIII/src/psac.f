      program psac
     
c-----
c     plot a trace at a given coordinate
c-----
c     Changes:
c     11 DEC 2018 - if the xmin xmax are defined, then the
c        plot will only put traces within these limits
c-----
      integer MAXPTS
      parameter (MAXPTS=8192)
      real y(MAXPTS), x(MAXPTS)
      character fname*80
      character cmd*10
      integer lgstr
      logical lsetymax, lsettwin
c-----
c     parse command line
c-----
      call gcmdln()
      call pinitf('PSAC.PLT')
      lsetymax = .false. 
      ymax = -12345
      lsettwin = .false.
      xfac = 1.0
 1000 continue
      read(5,*,end=2000)cmd,x0,y0,xlen,ylen,ipen,fname
      ls=lgstr(fname)
      write(6,*)cmd, x0,y0,xlen,ylen,ipen,fname(1:ls)
      if(cmd(1:5).eq.'TRACE')then
           call rsac1(fname,y,npts,btime,dt,MAXPTS,nerr)
           do i=1,npts
              x(i) = btime + (i-1)*dt
           enddo
           call doplotit(x0,y0,xlen,ylen,x,y,npts,ipen,ymax,lsetymax,
     1         tmin,tmax,lsettwin,btime)
      else if(cmd(1:6).eq.'CENTER')then
           call annotate('CENTER',x0,y0,ylen,ipen,fname,ls)
      else if(cmd(1:4).eq.'LEFT')then
           call annotate('LEFT',x0,y0,ylen,ipen,fname,ls)
      else if(cmd(1:3).eq.'XN')then
           xfac = ylen
      else if(cmd(1:4).eq.'YMAX')then
           lsetymax = .true.
           ymax = ylen
           WRITE(6,*)'ymax set to ',ymax
      else if(cmd(1:5).eq.'WIDTH')then
           call gwidth(x0)
      else if(cmd(1:5).eq.'XAXIS')then
        read(5,*)tmin,tmax
        lsettwin = .true.
        call dolinx(x0,y0,xlen,tmax,tmin,
     1      0.15*xfac,.true.,.false.,.true.,ls,fname)
c-----
c       put in tic marks along the x-axis 
c
c       x0  R*4 - position of left side of axis
c       y0  R*4 - position of left side of axis
c       xleng   R*4 - length of X axis
c       xmax    R*4 - maximum value of number corresponding to
c                   far right
c       xmin    R*4 - maximum value of number corresponding to
c                   far left
c       sizex   R*4 - size of numbers, e.g., 10
c       ticup   L   - .true. tics go up
c                 .false. tics go down
c       labtop  L   - .true. number goes above axis
c                 .false. goes below
c       dopow   L   - .true. put in number labels
c       llx I*4 length of x-axis title string
c       titlex  Ch  x-axis title string
      endif
      go to 1000
 2000 continue
      call pend()
      end

      subroutine gcmdln()
      integer mnmarg
      integer i, nmarg
      character*50 name
      nmarg = mnmarg()
      i = 0
 1000 continue
      i = i + 1
      if(i.gt.nmarg)go to 2000
      call mgtarg(i,name)
      if(name(1:2).eq.'-h')then
          call usage()
      endif
      go to 1000
 2000 continue
      end

      subroutine usage()
      integer LER
      parameter (LER=0)
      write(LER,*)'Usage: psac [-h]'
      write(LER,*)'Read input lines from standard input of form:'
      WRITE(LER,*)
     1'  command x0  y0 xlen ylen ipen fname'
      WRITE(LER,*)
     1'      where command and fname single quote delimited strings'
      WRITE(LER,*)
     1'      x0, y0, xlen and ylen are floats; ipen is integer'
      WRITE(LER,*)
     1'  Commands are TRACE, XAXIS, CENTER, LEFT, XF and WIDTH'
      WRITE(LER,*)
     1'  TRACE - plot a trace from (x0,y0) of length xlen, maximum ',
     2' amplitude is ylen. ipen is the pen color, fname is Sac file.'
      WRITE(LER,*)
     1'  XAXIS - plot time axis starting at (x0,y0) of lentch xlen',
     2' ylen and ipen not used, fname is axis title. '
      WRITE(LER,*)
     1'   NOTE: the next two lines of input are tmin and tmax for axis.'
      WRITE(LER,*)
     1'  CENTER - plot a string centered at (x0,y0) with height',
     2' ylen with color ipen, fname is string'
      WRITE(LER,*)
     1'  LEFT  - plot a string left jsutified at (x0,y0) with height',
     2' ylen with color ipen, fname is string'
      WRITE(LER,*)
     1'  XF - increase font size of XAXIS string by a factor xlen.'
      WRITE(LER,*)
     1'  WIDTH -change width of trace line to x0. The default is 0.001'
      WRITE(LER,*)
     1'  YMAX - plot absolute amplitude using the value in ylen'
      stop
      end


      subroutine annotate(type,xx,yy,ht,ipen,str,ls)
      real x0,y0,ht
      integer ipen, ls
      character str*(*), type*(*)
      call newpen(1)
      angle = 0.0
      if(type.eq.'LEFT')then
        call gleft(xx,yy,ht,str(1:ls),angle)
      else if(type.eq.'CENTER')then
        call gcent(xx,yy,ht,str(1:ls),angle)
      endif
      return
      end

      subroutine doplotit(x0,y0,xlen,ylen,x,y,npts,ipen,ymax,lsetymax,
     1   tmin,tmax,lsettwin,btime)
      integer npts
      real x0, y0, xlen, ylen 
      real y(npts), x(npts)
      logical lsetymax, lsettwin
      real tmin, tmax
      call scmxmn(y,npts,depmax,depmin,depmen,indmax,indmin)
      WRITE(6,*)'lsetymax:',lsetymax
      if(lsetymax)then
      WRITE(6,*)'using ',ymax,'trace has',amax1(abs(depmax),abs(depmin))
      else
      ymax = amax1(abs(depmax),abs(depmin))
      WRITE(6,*)'Using ',ymax
      endif
      WRITE(6,*)ymax
      call newpen(ipen)
      do i=1,npts
        if(lsettwin)then
        xx = x0 + xlen*(x(i) - tmin)/(tmax-tmin)
        else
        xx = x0 + (i-1)*xlen/(npts-1)
        endif
        yy = y0 + 0.5*ylen*y(i)/ymax
        if(i.eq.1)then
           call plot(xx,yy,3)
        else if(i.eq.npts)then
           call plot(xx,yy,2)
           call plot(xx,yy,3)
        else
           call plot(xx,yy,2)
        endif
          
         
      enddo
        call newpen(1)
      return
      end
