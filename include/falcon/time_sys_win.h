/*
   FALCON - The Falcon Programming Language.
   FILE: time_sys_win.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Win specific system time definitions
*/

#ifndef flc_time_sys_win_H
#define flc_time_sys_win_H

#include <windows.h>
#include <falcon/time_sys.h>

namespace Falcon {

class WinSystemTime: public SystemTime
{
public:
   SYSTEMTIME m_time;

   WinSystemTime( SYSTEMTIME val ):
      m_time( val )
   {
   }
};

}

#endif

/* end of time_sys_win.h */
