/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: heap_windows.cpp
   $Id: heap_windows.cpp,v 1.1.1.1 2006/10/08 15:05:20 gian Exp $

   Initialization of heap_windows variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-11-01 10:20+0100UTC
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Implementation of windows specific heap.
*/

#include <falcon/heap.h>

namespace Falcon
{
   HANDLE HeapMem::m_heapHandle = 0;
}


/* end of heap_windows.cpp */
