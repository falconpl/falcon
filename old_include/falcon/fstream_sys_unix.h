/*
   FALCON - The Falcon Programming Language.
   FILE: file_srv_unix.h

   UNIX system specific data used by FILE service.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   UNIX system specific data used by file streams service.
*/

#ifndef flc_fstream_sys_unix_H
#define flc_fstream_sys_unix_H

#include <falcon/fstream.h>

namespace Falcon {

/** Unix specific stream service support.
   This class provides UNIX system specific data to FILE service.
*/
class UnixFileSysData: public FileSysData
{
public:
   int m_handle;
   int m_lastError;

   UnixFileSysData( int handle, int m_lastError ):
      m_handle( handle ),
      m_lastError( m_lastError )
   {}

   virtual FileSysData *dup();
};

}

#endif

/* end of file_srv_unix.h */
