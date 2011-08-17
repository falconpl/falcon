/*
   FALCON - The Falcon Programming Language.
   FILE: generatorseq.cpp

   Virtual sequence that can be used to iterate over generators.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Aug 2009 22:10:00 +020

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/generatorseq.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>
#include <falcon/iterator.h>
#include <falcon/corerange.h>
#include <falcon/mempool.h>
#include <falcon/vm.h>

namespace Falcon {

GeneratorSeq::GeneratorSeq( VMachine* runEvn, const Item& callable ):
   m_vm( runEvn ),
   m_callable( callable ),
   m_bHasCachedCur( false ),
   m_bHasCachedNext( false ),
   m_bComplete( false )
{}

GeneratorSeq::GeneratorSeq( const GeneratorSeq& other ):
   m_vm( other.m_vm ),
   m_callable(other.m_callable),
   m_cache_cur( other.m_cache_cur ),
   m_cache_next( other.m_cache_next ),
   m_bHasCachedCur( other.m_bHasCachedCur ),
   m_bHasCachedNext( other.m_bHasCachedNext ),
   m_bComplete( other.m_bComplete )
{}

GeneratorSeq::~GeneratorSeq()
{}


GeneratorSeq* GeneratorSeq::clone() const
{
   return new GeneratorSeq( *this );
}

const Item &GeneratorSeq::front() const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "GeneratorSeq::front" ) );
}

const Item &GeneratorSeq::back() const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "GeneratorSeq::back" ) );
}


void GeneratorSeq::clear()
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "GeneratorSeq::clear" ) );
}

bool GeneratorSeq::empty() const
{
   if ( m_bComplete )
      return true;

   if ( m_bHasCachedCur )
      return false;

   return ! fillCurrentValue();
}

void GeneratorSeq::gcMark( uint32 gen )
{
   memPool->markItem( m_callable );

   if( m_bHasCachedCur )
      memPool->markItem( m_cache_cur );
   if ( m_bHasCachedNext )
      memPool->markItem( m_cache_next );

   Sequence::gcMark( gen );
}


bool GeneratorSeq::fillCurrentValue() const
{
   if( m_bComplete )
      return false;

   m_vm->callItemAtomic( m_callable, 0 );
   m_cache_cur = m_vm->regA();
   if( m_cache_cur.isOob() && m_cache_cur.isInteger() && m_cache_cur.asInteger() == 0 )
   {
     m_bComplete = true;
     m_bHasCachedCur = false;
     return false;
   }

   m_bHasCachedCur = true;
   return true;
}

bool GeneratorSeq::fillNextValue() const
{
   if( m_bComplete )
      return false;

   m_vm->callItemAtomic( m_callable, 0 );
   m_cache_next = m_vm->regA();
   if( m_cache_next.isOob() && m_cache_next.isInteger() && m_cache_next.asInteger() == 0 )
   {
      m_bHasCachedNext = false;
      m_bComplete = true;
      return false;
   }
   m_bHasCachedNext = true;
   return true;
}


void GeneratorSeq::append( const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "GeneratorSeq::append" ) );
}

void GeneratorSeq::prepend( const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "GeneratorSeq::prepend" ) );
}

void GeneratorSeq::getIterator( Iterator& tgt, bool tail ) const
{
   Sequence::getIterator( tgt, tail );
}

void GeneratorSeq::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
}

void GeneratorSeq::insert( Iterator &, const Item & )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "GeneratorSeq::insert" ) );
}

void GeneratorSeq::erase( Iterator & )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "GeneratorSeq::erase" ) );
}


bool GeneratorSeq::hasNext( const Iterator &iter ) const
{
   if( ! m_bHasCachedCur )
   {
      if( ! fillCurrentValue() )
         return false;
   }

   if ( ! m_bHasCachedNext )
   {
      if( ! fillNextValue() )
         return false;
   }

   return true;
}

bool GeneratorSeq::hasPrev( const Iterator &iter ) const
{
   return true; // will eventually raise on prev.
}

bool GeneratorSeq::hasCurrent( const Iterator &iter ) const
{
   if( ! m_bHasCachedCur )
   {
      if( ! fillCurrentValue() )
         return false;
   }

   return true;
}

bool GeneratorSeq::next( Iterator &iter ) const
{
   // if we don't have a current value anymore, cache it.
   if( ! m_bHasCachedCur )
   {
      if( ! fillCurrentValue() )
         return false;

      m_bHasCachedNext = false;
      return true;
   }
   else
   {
      // we have a current value; but have we got a next value?
      if( m_bHasCachedNext )
      {
         // Yes --  demote it to current.
         m_cache_cur = m_cache_next;
         m_bHasCachedNext = false;
         return true;
      }
      else {
         // no? -- absorb a next value.
         return fillCurrentValue();
      }
   }
}

bool GeneratorSeq::prev( Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "GeneratorSeq::prev" ) );
}

Item& GeneratorSeq::getCurrent( const Iterator &iter )
{
   if( ! m_bHasCachedCur )
   {
      fillCurrentValue();
      // TODO: raise if terminated? -- one should call next() or hasCurrent(), but..
   }

   return m_cache_cur;
}

Item& GeneratorSeq::getCurrentKey( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ )
            .origin( e_orig_runtime ).extra( "GeneratorSeq::getCurrentKey" ) );
}

bool GeneratorSeq::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return true;
}


}

/* end of generatorseq.cpp */
