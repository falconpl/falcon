/*
   FALCON - The Falcon Programming Language.
   FILE: callframe.cpp

   Closure - function and externally referenced local variabels
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
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/symbol.h>

#include <string.h>

namespace Falcon {

Closure::Closure( Class* handler, void *closed ):
   m_closed(closed),
   m_handler( handler ),
   m_closedData(0),
   m_closedDataSize(0),
   m_closedLocals(0),
   m_mark(0)
{
}

Closure::Closure():
   m_closed(0),
   m_handler(0),
   m_closedData(0),
   m_closedDataSize(0),
   m_closedLocals(0),
   m_mark(0)
{
}

Closure::Closure( const Closure& other )
{
   if( other.m_handler != 0 && other.m_closed != 0 ) {
      m_handler = other.m_handler;
      m_closed = m_handler->clone(other.m_closed);
      m_closedDataSize = other.m_closedDataSize;      
      m_closedData = new Variable[m_closedDataSize];
      m_closedLocals = other.m_closedLocals;
      // the variables in a closure are SURELY references, so we can flat-copy them.
      memcpy( m_closedData, other.m_closedData, m_closedDataSize * sizeof(Variable));
   }
   else {
      m_handler = 0;
      m_closed = 0;
      m_closedDataSize = 0;      
      m_closedData = 0;
      m_closedLocals = 0;
   }
}

Closure::~Closure()
{
   delete[] m_closedData;
}

void Closure::gcMark( uint32 mark )
{
   if( m_mark < mark )
   {
      m_mark = mark;
      m_handler->gcMark(mark);
      m_handler->gcMarkInstance(m_closed, mark);
      for( uint32 i = 0; i < m_closedDataSize; ++i ) {         
         m_closedData[i].gcMark( mark );
         m_closedData[i].value()->gcMark(mark);
      }
   }
}

uint32 Closure::pushClosedData( VMContext* ctx )
{
   for( uint32 i = 0; i < m_closedDataSize; ++ i ) 
   {
      ctx->pushData( *m_closedData[i].value() );
   }
   
   return m_closedDataSize;
}


void Closure::close( VMContext* ctx, const SymbolTable* st )
{
   fassert( m_closed != 0 );
   TRACE( "Closure::close %p", m_closed );

   delete[] m_closedData;
   uint32 size = st->closedCount();
   m_closedDataSize = size;
   m_closedData = new Variable[size];
   m_closedLocals = st->localCount();
   
   for( uint32 i = 0; i < size; ++i )
   {
      Symbol* closed = st->getClosed(i);
      TRACE1( "Closure::close -- closing symbol %s", closed->name().c_ize() );
      // navigate through the parent symbol tables till finding the desired symbols.
      long depth = ctx->callDepth();
      {
         long curDepth = 0;
         while( curDepth < depth )
         {
            const CallFrame& cf = ctx->previousFrame( curDepth );
            fassert( cf.m_function != 0 )
            Symbol* tgtsym = cf.m_function->symbols().findSymbol( closed->name() );
            if( tgtsym != 0 )
            {
               // we found it. get the item.
               Variable* variable = tgtsym->getVariable(ctx);
               fassert( variable != 0 );
               // now reference it in our closure array
               m_closedData[closed->localId()].makeReference(variable);
               // and since we're done, we can break;
               TRACE2( "Closure::close -- closed symbol %s at depth %d => \"%s\"",
                  closed->name().c_ize(), (int)curDepth, variable->value()->describe().c_ize() );
               break;
            }
            // better luck with next time.
            curDepth++;
         }
      }
      // if we didn't find the symbol, we just reference a nil -- that's ok
      // (weird, but ok)
   }
}

}

/* end of closure.cpp */
