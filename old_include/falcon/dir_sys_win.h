/*
   FALCON - The Falcon Programming Language.
   FILE: dir_win.h

   Support for directory oriented operations in unix systems
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Support for directory oriented operations in unix systems
*/

#ifndef flc_dir_win_H
#define flc_dir_win_H

#include <windows.h>
#ifndef INVALID_FILE_ATTRIBUTES
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#endif

#include <falcon/dir_sys.h>

namespace Falcon {

class String;

class DirEntry_win: public DirEntry
{

protected:
   HANDLE m_handle;
   WIN32_FIND_DATAW m_raw_dir;
   bool m_first;
   uint32 m_lastError;

public:
   DirEntry_win( const String &p, HANDLE handle, WIN32_FIND_DATAW dir_data ):
      DirEntry(p),
      m_first( true ),
      m_lastError( 0 ),
      m_handle( handle ),
      m_raw_dir( dir_data )
   {}

   ~DirEntry_win() {
      close();
   }

   virtual bool read( String &name );
   virtual void close();
};

}

#endif

/* end of dir_win.h */
