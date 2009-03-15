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
namespace Sys {

class WIN_TH
{
public:
   WIN_TH( HANDLE canc );
   ~WIN_TH();

   DWORD thID;
   int lastError;
   HANDLE hth;
   
   // will just point to the cancel handle of the VM where we've been creeated
   HANDLE eCancel;
};


}
}

#endif

/* end of systhread_win.h */
