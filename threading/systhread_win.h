/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: systhread_win.h

   System dependent MT provider - MS-Windows provider specific data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 09 Apr 2008 21:32:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependent MT provider - MS-Windows provider specific data.
*/

#ifndef FLC_SYSTHREAD_WIN_H
#define FLC_SYSTHREAD_WIN_H

#define _WIN32_WINNT 0x0403
#include <windows.h>
#include <falcon/setup.h>

namespace Falcon {
namespace Ext {

class WIN_THI_DATA
{
public:
   WIN_THI_DATA();
   ~WIN_THI_DATA();

   DWORD thID;
   int lastError;
   HANDLE hth;
};


}
}

#endif

/* end of systhread_win.h */
