/*
   FALCON - The Falcon Programming Language.
   FILE: dir_sys_unix.h
   $Id: dir_sys_unix.h,v 1.1 2007/06/21 21:54:25 jonnymind Exp $

   Support for directory oriented operations in unix systems
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004
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
   Support for directory oriented operations in unix systems.
*/

#ifndef flc_dir_sys_unix_H
#define flc_dir_sys_unix_H

#include <falcon/filestat.h>
#include <falcon/dir_sys.h>

#include <sys/types.h>
#include <dirent.h>


namespace Falcon {

class String;

/** Support for low level directory system for unix. */
class DirEntry_unix: public DirEntry
{

protected:
   DIR *m_raw_dir;

public:
   DirEntry_unix( DIR *d ):
      DirEntry(),
      m_raw_dir( d )
   {}

   ~DirEntry_unix() {
      close();
   }

   virtual bool read( String &dir );
   virtual void close();
};

}


#endif

/* end of dir_sys_unix.h */
