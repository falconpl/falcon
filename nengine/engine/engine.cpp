/*
   FALCON - The Falcon Programming Language
   FILE: engine.cpp

   Engine static/global data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 02 Mar 2009 20:33:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Engine static/global data setup and initialization
*/

#include <falcon/string.h>
#include <falcon/error_messages.h>
#include <falcon/mt.h>

namespace Falcon
{
#define FLC_DECLARE_ENGINE_MSG
#include <falcon/eng_messages.h>
#undef FLC_DECLARE_ENGINE_MSG

namespace Engine
{
   static Mutex s_mtx;

#ifdef FALCON_SYSTEM_WIN
   static bool s_bWindowsNamesConversion = true;
#else
   static bool s_bWindowsNamesConversion = false;
#endif

}

}

/* end of engine.cpp */
