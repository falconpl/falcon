/*
   FALCON - The Falcon Programming Language.
   FILE: callframe.cpp

   Closure - function and externally referenced local variables
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 15:39:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/closure.cpp"

#include <falcon/trace.h>

#include <falcon/closure.h>
#include <falcon/item.h>
#include <falcon/vmcontext.h>
#include <falcon/callframe.h>
#include <falcon/mt.h>
#include <falcon/function.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>
#include <falcon/symbol.h>

#include <map>
#include <vector>

#include <string.h>

namespace Falcon {


Closure::Closure():
   m_mark(0),
   m_closed(0),
   m_data(0)
{
}

Closure::Closure( Function* f ):
   m_mark(0),
   m_closed(f),
   m_data(0)
{
}

Closure::Closure( const Closure& other ):
   m_mark(0),
   m_closed( other.m_closed ),
   m_data( other.m_data )
{
}

Closure::~Closure()
{
}

void Closure::gcMark( uint32 mark )
{
   if( m_mark == mark ) {
      return;
   }
   m_mark = mark;

   if ( m_closed != 0 ) {
      m_closed->gcMark(mark);
   }
   if( m_data != 0 ) {
      m_data->gcMark(mark);
   }
}


void Closure::flatten( VMContext*, ItemArray& subItems ) const
{
   if( m_closed != 0 ) {
      subItems.append( m_closed );
   }
   else {
      subItems.append( Item() );
   }

   if( m_data != 0 ) {
      subItems.append( Item( m_data->handler(), m_data ) );
   }
   else {
      subItems.append( Item() );
   }
}


void Closure::unflatten( VMContext*, ItemArray& subItems, uint32 pos )
{
   if( pos +2 >= subItems.length() ) {
      return;
   }

   Item& closedItem = subItems[pos++];
   Item& dataItem = subItems[pos++];

   m_closed = closedItem.isNil() ? 0 : closedItem.asFunction();
   m_data = dataItem.isNil() ? 0 : static_cast<ClosedData*>(dataItem.asInst());
}


const Class* Closure::handler() const
{
   static const Class* cls = Engine::handlers()->closureClass();
   return cls;
}


/** Analyzes the function and the context and closes the needed values.
 \param ctx the context where the closed data is to be found.
 \param A symbol table containing the symbols to be closed.
 */
void Closure::close( VMContext* ctx )
{
   const SymbolMap& vars = m_closed->closed();
   TRACE( "Closure::close %s -- %d vars", m_closed->name().c_ize(), vars.size() )

   uint32 closedCount = vars.size();
   if( closedCount == 0 ) return;

   CallFrame* current = &ctx->currentFrame();
   ClosedData* cd = ctx->getTopClosedData();
   if( cd != 0 ) {
      if( cd == current->m_closingData ) {
         m_data = current->m_closingData;
      }
      else {
         if( m_data == 0 ) {
            m_data = new ClosedData;
            FALCON_GC_HANDLE( m_data );
         }
         m_data->copy( *cd );
      }
   }
   else {
      if( m_data == 0 ) {
         m_data = new ClosedData;
         FALCON_GC_HANDLE( m_data );
      }
   }

   current->m_closingData = m_data;

   for( uint32 i = 0; i < closedCount; ++i )
   {
      const Symbol* sym = vars.getById(i);
      Item* data = ctx->resolveSymbol( sym, false );

      if( data != 0 )
      {
         TRACE1( "Closure::close %s -- found %s", m_closed->name().c_ize(), sym->name().c_ize() );
         m_data->add(sym, *data);
      }
      else {
         TRACE1( "Closure::close %s -- NOT found %s", m_closed->name().c_ize(), sym->name().c_ize() );
      }
   }
}

}

/* end of closure.cpp */
