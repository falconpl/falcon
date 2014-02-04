/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: iterator.cpp
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#include "iterator_mod.h"
#include "container_mod.h"

namespace Falcon {
namespace Mod {

Iterator::~Iterator()
{}

void Iterator::gcMark( uint32 m )
{
   if( m_mark != m )
   {
      m_mark = m;
      container()->gcMark(m);
   }
}
}

}

/* end of iterator.h */

