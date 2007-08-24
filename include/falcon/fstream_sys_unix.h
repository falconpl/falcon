/*
   FALCON - The Falcon Programming Language.
   FILE: file_srv_unix.h
   $Id: fstream_sys_unix.h,v 1.2 2007/08/03 13:17:05 jonnymind Exp $

   UNIX system specific data used by FILE service.
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
