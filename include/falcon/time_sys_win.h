/*
   FALCON - The Falcon Programming Language.
   FILE: time_sys_win.h
   $Id: time_sys_win.h,v 1.2 2007/06/22 15:14:24 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
