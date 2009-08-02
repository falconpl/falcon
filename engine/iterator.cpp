/*
   FALCON - The Falcon Programming Language.
   FILE: iterator.cpp

   Implementation of virtual functions for iterators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Aug 2009 13:00:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/iterator.h>

namespace Falcon {

Iterator::~Iterator()
{
   m_owner->disposeIterator( *this );
}

void Iterator::gcMark( uint32 mark )
{
   m_owner->gcMarkIterator( *this );
   m_owner->gcMark( mark );
}

FalconData *Iterator::clone() const
{
   return new Iterator( *this );
}

}

/* end of iterator.cpp */
