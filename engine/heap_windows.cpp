/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: heap_windows.cpp

   Initialization of heap_windows variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-11-01 10:20+0100UTC

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
