/*
   FALCON - The Falcon Programming Language
   FILE: poopseq.h

   Virtual sequence that can be used to iterate over poop providers.
   *AT THE MOMENT* providing just "append" method to re-use sequence
   comprehension in OOP and POOP contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 10 Aug 2009 10:53:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/poopseq.h>
#include <falcon/rosstream.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>
#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/item.h>

namespace Falcon {

Item m_appendMth;
uint32 m_mark;

PoopSeq::PoopSeq( VMachine* vm, const Item &iobj ):
   m_mark( vm->generation() ),
   m_vm( vm )
{
   switch( iobj.type() )
   {
   case FLC_ITEM_OBJECT:
      if( ! iobj.asObjectSafe()->getMethod( "append", m_appendMth ) )
      {
         throw new CodeError( ErrorParam( e_miss_iface, __LINE__ )
              .origin( e_orig_runtime )
              .extra( "append") );
      }
      break;

   case FLC_ITEM_DICT:
      if( ! iobj.asDict()->getMethod( "append", m_appendMth ) )
      {
         throw new CodeError( ErrorParam( e_miss_iface, __LINE__ )
              .origin( e_orig_runtime )
              .extra( "append") );
      }
      break;

   default:
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
           .origin( e_orig_runtime )
           .extra( "O|D" ) );
   }
}

PoopSeq::PoopSeq( const PoopSeq& other ):
   m_appendMth( other.m_appendMth ),
   m_mark( other.m_vm->generation() ),
   m_vm( other.m_vm )
{}


PoopSeq::~PoopSeq()
{
}


bool PoopSeq::empty() const
{
   return true;
}

void PoopSeq::gcMark( uint32 gen )
{
   if ( m_mark != gen )
   {
      m_mark = gen;
      Sequence::gcMark( gen );
      memPool->markItem( m_appendMth );
   }
}


PoopSeq* PoopSeq::clone() const
{
   return new PoopSeq( *this );
}


const Item &PoopSeq::front() const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
      .origin( e_orig_runtime )
      .extra( "PoopSeq::front" ) );
}

const Item &PoopSeq::back() const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "PoopSeq::back" ) );
}

void PoopSeq::clear()
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "PoopSeq::clear" ) );
}

void PoopSeq::append( const Item& itm )
{
   m_vm->pushParam( itm );
   m_vm->callItemAtomic( m_appendMth, 1 );
}

void PoopSeq::prepend( const Item& itm )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ ).origin( e_orig_runtime ).extra( "PoopSeq::prepend" ) );
}


//============================================================
// Iterator management.
//============================================================

void PoopSeq::getIterator( Iterator& tgt, bool tail ) const
{
   Sequence::getIterator( tgt, tail );
   // nothing to do
}


void PoopSeq::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
   // nothing to do
}


void PoopSeq::insert( Iterator &iter, const Item &data )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "PoopSeq::insert" ) );
}

void PoopSeq::erase( Iterator &iter )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "PoopSeq::erase" ) );
}


bool PoopSeq::hasNext( const Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "PoopSeq::erase" ) );
}


bool PoopSeq::hasPrev( const Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "PoopSeq::erase" ) );
}

bool PoopSeq::hasCurrent( const Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "PoopSeq::next" ) );
}


bool PoopSeq::next( Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
         .origin( e_orig_runtime ).extra( "PoopSeq::next" ) );
}


bool PoopSeq::prev( Iterator &iter ) const
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
            .origin( e_orig_runtime ).extra( "PoopSeq::prev" ) );
}

Item& PoopSeq::getCurrent( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_not_implemented, __LINE__ )
               .origin( e_orig_runtime ).extra( "PoopSeq::prev" ) );
}


Item& PoopSeq::getCurrentKey( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ )
      .origin( e_orig_runtime ).extra( "PoopSeq::getCurrentKey" ) );
}


bool PoopSeq::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return false;
}


}

/* end of poopseq.cpp */
