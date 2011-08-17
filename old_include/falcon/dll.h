/*
   FALCON - The Falcon Programming Language.
   FILE: flc_dll.h

   Base class for Dynamic load system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_DLL_H
#define flc_DLL_H

#if defined(FALCON_SYSTEM_WIN)
	#include <falcon/dll_win.h>
#elif defined(FALCON_SYSTEM_MAC)
	#include <falcon/dll_mac.h>
#else
	#include <falcon/dll_dl.h>
#endif

#endif
/* end of flc_dll.h */
