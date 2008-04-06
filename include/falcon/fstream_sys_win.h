/*
   FALCON - The Falcon Programming Language.
   FILE: fstream_sys_win.h

   MS-Windows system specific data used by FILE service.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   MS-Windows system specific data used by FILE service.
*/

#ifndef flc_fstream_sys_win_H
#define flc_fstream_sys_win_H

#include <falcon/fstream.h>
#include <windows.h>
#include <stdlib.h>

namespace Falcon {

/** Win specific stream service support.
   This class provides MS-Windows system specific data to FILE service.
*/
class FALCON_DYN_CLASS WinFileSysData: public FileSysData
{
public:
   typedef enum {
      e_dirIn,
      e_dirOut
   } t_direction;

   HANDLE m_handle;
   DWORD m_lastError;

   bool m_isConsole;
   bool m_isPipe;
   t_direction m_direction;

   WinFileSysData( HANDLE handle, DWORD m_lastError, bool console=false, t_direction dir=e_dirIn, bool pipe = false ):
      m_handle( handle ),
      m_lastError( m_lastError ),
      m_isConsole( console ),
      m_direction( dir ),
      m_isPipe( pipe )
   {}

   virtual FileSysData *dup();
};

}

#endif

/* end of fstream_sys_win.h */
