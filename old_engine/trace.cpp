/*
   FALCON - The Falcon Programming Language.
   FILE: trace.cpp

   Debug trace utility.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 18 Jul 2009 14:07:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef NDEBUG

#include <falcon/trace.h>
FALCON_DYN_SYM _falcon_trace_level >= 1 &&FILE* _falcon_trace_fp = 0;

#endif

/* end of trace.cpp */
