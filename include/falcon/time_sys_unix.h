/*
   FALCON - The Falcon Programming Language.
   FILE: time_sys_unix.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
