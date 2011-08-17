/*
   FALCON - The Falcon Programming Language.
   FILE: dir_sys_unix.h

   Support for directory oriented operations in unix systems
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
   DirEntry_unix( const String &p, DIR *d ):
      DirEntry(p),
      m_raw_dir( d )
   {}

   virtual ~DirEntry_unix() {
      close();
   }

   virtual bool read( String &dir );
   virtual void close();
};

}


#endif

/* end of dir_sys_unix.h */
