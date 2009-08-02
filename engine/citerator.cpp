/*
   FALCON - The Falcon Programming Language.
   FILE: citerator.cpp

   Base abstract class for generic collection iterators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 27 Mar 2009 01:45:22 -0700

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Base abstract class for generic collection iterators.
   Implementation of non virtual functions
*/

#include <falcon/citerator.h>
#include <falcon/item.h>
#include <falcon/mempool.h>
#include <falcon/sequence.h>

namespace Falcon {

CoreIterator::CoreIterator():
   m_creator(0),
   m_creatorSeq(0)
{
}

CoreIterator::CoreIterator( const CoreIterator& other ):
   m_creator(0),
   m_creatorSeq(0)
{
   m_creator = other.m_creator;
   m_creatorSeq = other.m_creatorSeq;
}


CoreIterator::~CoreIterator()
{
}


void CoreIterator::setOwner( Garbageable *owner )
{
   m_creator = owner;
}

void CoreIterator::setOwner( Sequence *owner )
{
   m_creatorSeq = owner;
}

void CoreIterator::gcMark( uint32 mark )
{
   // AARRG
   if ( m_creator != 0 )
      m_creator->mark( mark );

   if ( m_creatorSeq != 0 )
      m_creatorSeq->gcMark( mark );
}

}

/* end of citerator.cpp */
