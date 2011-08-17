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
   // give the owner a chance to de-account us
   if( m_owner != 0 )
      m_owner->disposeIterator( *this );

   // Then, get rid of deep data, if we have to
   if ( m_deletor != 0 )
      m_deletor( this );
}

void Iterator::gcMark( uint32 mark )
{
   if( m_owner )
   {
      m_owner->gcMark( mark );
   }
}

Iterator *Iterator::clone() const
{
   return new Iterator( *this );
}



}

/* end of iterator.cpp */
