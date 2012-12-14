/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: fstream_win.h

   WINDOWS system specific FILE service support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin:

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Widnows system specific FILE service support.
*/

#ifndef _FALCON_FSTREAM_WIN_H_
#define _FALCON_FSTREAM_WIN_H_

#include <falcon/setup.h>
#include <windows.h>

namespace Falcon {

class WinFStreamData {
public:
   HANDLE hFile;
   bool bIsFile;
   bool bNonBlocking;

   WinFStreamData( HANDLE hf, bool bf = true, bool nb = false ):
      hFile( hf ),
      bIsFile( bf ),
      bNonBlocking( nb )
   {}
};

}

#endif

/* end of fstream_win.h */
