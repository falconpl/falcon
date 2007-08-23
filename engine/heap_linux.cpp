/*
   FALCON - The Falcon Programming Language.
   FILE: heap_linux.cpp
   $Id: heap_linux.cpp,v 1.1.1.1 2006/10/08 15:05:19 gian Exp $

   Initialization of heap_linux variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio set 30 2004
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
   Short description
*/

#include <falcon/heap.h>

namespace Falcon
{
   long HeapMem::m_pageSize = 0;
}


/* end of heap_linux.cpp */
