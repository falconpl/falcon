/*
   FALCON - The Falcon Programming Language.
   FILE: time_sys_unix.h
   $Id: time_sys_unix.h,v 1.1 2007/06/21 21:54:26 jonnymind Exp $

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
   Unix specific system time definitions
*/

#ifndef flc_time_sys_unix_H
#define flc_time_sys_unix_H

#include <time.h>
#include <falcon/time_sys.h>

namespace Falcon {

class UnixSystemTime: public SystemTime
{
public:
   time_t m_time_t;

   UnixSystemTime( time_t val ):
      m_time_t( val )
   {
   }
};

}

#endif

/* end of time_sys_unix.h */
