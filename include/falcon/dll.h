/*
   FALCON - The Falcon Programming Language.
   FILE: flc_dll.h
   $Id: dll.h,v 1.1.1.1 2006/10/08 15:05:29 gian Exp $

   Base class for Dynamic load system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/


#ifndef flc_DLL_H
#define flc_DLL_H

#ifdef FALCON_SYSTEM_WIN
#include <falcon/dll_win.h>
#elif FALCON_SYSTEM_MAC
#include <falcon/dll_mac.h>
#else
#include <falcon/dll_dl.h>
#endif

#endif
/* end of flc_dll.h */
