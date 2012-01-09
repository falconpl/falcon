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

namespace Falcon {

Closure::Closure( Function* func ):
   m_function(func)
{
   // it's pretty stupid to create a closure if you don't have closed symbols, but...
   uint32 size = func->symbols().closedCount();
   m_closedData.resize(size);
}


Closure::Closure( const Closure& other ):
   m_function( other.m_function ),
   m_closedData( other.closedData() )
{
}

Closure::~Closure()
{
}

void Closure::gcMark( uint32 mark )
{
   if( m_mark < mark )
   {
      m_mark = mark;
      m_function->gcMark( mark );
      m_closedData.gcMark( mark );
   }
}


void Closure::close( VMContext* ctx )
{
   fassert( m_function != 0 );
   TRACE( "Closure::close %s", m_function->name().c_ize() );

   SymbolTable& symtab = m_function->symbols();
   uint32 size = symtab.closedCount();
   for( uint32 i = 0; i < size; ++i )
   {
      Symbol* closed = symtab.getClosed(i);
      TRACE1( "Closure::close -- closing symbol %s", closed->name().c_ize() );
      // navigate through the parent symbol tables till finding the desired symbols.
      long depth = ctx->callDepth();
      {
         long i = 0;
         while( i < depth )
         {
            const CallFrame& cf = ctx->previousFrame( i );
            fassert( cf.m_function != 0 )
            Symbol* tgtsym = cf.m_function->symbols().findSymbol( closed->name() );
            if( tgtsym != 0 )
            {
               // we found it. get the item.
               Item* theItem = tgtsym->getValue(ctx);
               fassert( theItem != 0 );
               // now reference it in our closure array
               ItemReference::create(*theItem, m_closedData[i]);
               // and since we're done, we can break;
               TRACE2( "Closure::close -- closed symbol %s at depth %d => \"%s\"",
                  closed->name().c_ize(), (int)i, theItem->describe().c_ize() );
               break;
            }
            // better luck with next time.
            ++i;
         }
      }
      // if we didn't find the symbol, we just reference a nil -- that's ok
      // (weird, but ok)
   }
}


// can be used only by ClassClosure
void Closure::function( Function* func )
{
   // don't care about previous values; this is a function internally used by ClassClosure
   m_function = func;
   // it's pretty stupid to create a closure if you don't have closed symbols, but...
   uint32 size = func->symbols().closedCount();
   m_closedData.resize( size );
}

}

/* end of closure.cpp */
