/*
   FALCON - The Falcon Programming Language.
   FILE: rangeseq.cpp

   Virtual sequence that can be used to iterate over ranges.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Aug 2009 22:10:00 +020

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/rangeseq.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>
#include <falcon/iterator.h>
#include <falcon/corerange.h>

namespace Falcon {

RangeSeq::RangeSeq( const CoreRange &rng )
{
   if ( rng.isOpen() )
      throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "open range") );

   m_start = rng.start();
   m_end = rng.end();
   m_step = rng.step();

   // force the step to be non-default
   if( m_step == 0 )
      m_step = m_start <= m_end ? 1 : -1;
}

RangeSeq::RangeSeq( int64 s, int64 e, int64 step ):
   m_start(s),
   m_end(e),
   m_step(step)
{
   // force the step to be non-default
   if( m_step == 0 )
      m_step = m_start <= m_end ? 1 : -1;
}

RangeSeq::~RangeSeq()
{}

RangeSeq* RangeSeq::clone() const
{
   return new RangeSeq( *this );
}

const Item &RangeSeq::front() const
{
   m_number = m_start;
   return m_number;
}

const Item &RangeSeq::back() const
{
   m_number = m_end;
   return m_number;
}


void RangeSeq::clear()
{
   m_start = m_end;
   m_step = 1;
}

bool RangeSeq::empty() const
{
   return m_step > 0 ? m_start >= m_end : m_start < m_end;
}

void RangeSeq::append( const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "RangeSeq::append" ) );
}

void RangeSeq::prepend( const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "RangeSeq::prepend" ) );
}

void RangeSeq::getIterator( Iterator& tgt, bool tail ) const
{
   Sequence::getIterator( tgt, tail );
   tgt.position( m_start );
}

void RangeSeq::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
   tgt.position( source.position() );
}

void RangeSeq::insert( Iterator &, const Item & )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "RangeSeq::insert" ) );
}

void RangeSeq::erase( Iterator & )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "RangeSeq::erase" ) );
}


bool RangeSeq::hasNext( const Iterator &iter ) const
{
   return m_step > 0 ?
         iter.position() + m_step < m_end :
         iter.position() + m_step >= m_end;
}

bool RangeSeq::hasPrev( const Iterator &iter ) const
{
   return m_step > 0 ?
         iter.position() - m_step > m_start :
         iter.position() - m_step < m_end;

}

bool RangeSeq::hasCurrent( const Iterator &iter ) const
{
   return m_step > 0 ?
         iter.position() < m_end :
         iter.position() >= m_end;
}

bool RangeSeq::next( Iterator &iter ) const
{
   if( m_step > 0 )
   {
      if ( iter.position() < m_end )
         iter.position( iter.position() + m_step );
      return iter.position() < m_end;
   }
   else
   {
      if ( iter.position() >= m_end )
         iter.position( iter.position() + m_step );
      return iter.position() >= m_end;
   }
}

bool RangeSeq::prev( Iterator &iter ) const
{
   if( m_step > 0 )
   {
      if ( iter.position() > m_start )
      {
         iter.position( iter.position() - m_step );
         return true;
      }
      else {
         // move past end
         if( (m_end - m_start) % m_step == 0 )
            iter.position( m_end );
         else
            iter.position( ((m_end - m_start)/ m_step + 1 )* m_step + m_start);

         return false;
      }
   }
   else
   {
      if ( iter.position() < m_start )
      {
         iter.position( iter.position() + m_step );
         return true;
      }
      else
      {
         iter.position( ((m_start - m_end)/ m_step - 1 )* m_step + m_start);
         return false;
      }
   }
}

Item& RangeSeq::getCurrent( const Iterator &iter )
{
   m_number = iter.position();
   return m_number;
}

Item& RangeSeq::getCurrentKey( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ )
            .origin( e_orig_runtime ).extra( "RangeSeq::getCurrentKey" ) );
}

bool RangeSeq::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return first.position() == second.position();
}

}

/* end of rangeseq.cpp */
