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
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/symbol.h>
#include <falcon/varmap.h>
#include <falcon/itemarray.h>

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
      m_closedData = new ItemArray();
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

}

void Closure::gcMark( uint32 mark )
{
   if( m_mark < mark )
   {
      m_mark = mark;
      m_handler->gcMark(mark);
      m_handler->gcMarkInstance(m_closed, mark);
      m_closedData->gcMark(mark);
   }
}

uint32 Closure::pushClosedData( VMContext* ctx )
{
   for( uint32 i = 0; i < m_closedDataSize; ++ i ) 
   {
      ctx->pushData( (*m_closedData)[i] );
   }
   
   return m_closedDataSize;
}


void Closure::close( VMContext* , const VarMap* st )
{
   fassert( m_closed != 0 );
   TRACE( "Closure::close %p", m_closed );

   delete[] m_closedData;
   uint32 size = st->closedCount();
   m_closedDataSize = size;
   m_closedLocals = st->localCount();
   
   for( uint32 i = 0; i < size; ++i )
   {
      const String& closedName = st->getClosedName(i);
      
      TRACE1( "Closure::close -- closing symbol %s", closedName.c_ize() );
      
      /*
       * TODO
       */
      /*
      Variable* variable = ctx-> ( closedName );
      if( variable != 0 ) {
         TRACE2( "Closure::close -- closed symbol %s => \"%s\"",
            closed->name().c_ize(), variable->value()->describe().c_ize() );

         m_closedData[closed->localId()].makeReference(variable);
      }
      else {
         TRACE2( "Closure::close -- didn't find symbol to close %s", closed->name().c_ize() );
         Variable::makeFreeVariable(m_closedData[closed->localId()]);
      }
      */
   }
}

}

/* end of closure.cpp */
