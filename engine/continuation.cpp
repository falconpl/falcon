/*
   FALCON - The Falcon Programming Language.
   FILE: continuation.h

   Continuation object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Dec 2009 17:04:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vm.h>
#include <falcon/continuation.h>

namespace Falcon
{

Continuation::Continuation( VMachine* vm ):
   m_vm( vm ),
   m_tgtSymbol(0),
   m_tgtLModule(0),
   m_stackBase(0),
   m_bPhase(false)
{
   m_context = vm->currentContext();
}


Continuation::Continuation( const Continuation& e ):
   m_vm( e.m_vm ),
   m_stack( e.m_stack ),
   m_tgtSymbol( e.m_tgtSymbol ),
   m_tgtLModule( e.m_tgtLModule ),
   m_bPhase( false )
{
   m_context = e.m_context;
   m_stackLevel = e.m_stackLevel;
   m_tgtPC = e.m_tgtPC;

}


Continuation::~Continuation()
{

}


void Continuation::callMark()
{
   m_stackLevel = m_vm->stackBase() - m_vm->currentFrame()->m_param_count -VM_FRAME_SPACE;
   // active phase
   m_bPhase = true;
}


bool Continuation::jump()
{
   if ( m_tgtSymbol != 0 )
   {
      // remove our frame, or we'll be called twice.
      m_vm->stack().copyOnto( m_stackLevel, m_stack, 0, m_stack.length() );
      m_vm->currentContext()->symbol( m_tgtSymbol );
      m_vm->currentContext()->lmodule( m_tgtLModule );
      m_vm->currentContext()->pc_next() = m_tgtPC;
      m_vm->currentContext()->stackBase() = m_stackBase;
      return true;
   }

   return false;
}


bool Continuation::unroll( VMachine* vm )
{
   // Unroll the stack
   /*while( m_vm->stackBase() > m_stackLevel )
   {
      // neutralize post-processors
      m_vm->returnHandler( 0 );
      m_vm->callReturn();
      if ( m_vm->breakRequest() )
      {
         // exit from the C frame, but be sure to be called back
         m_vm->returnHandler( Continuation::unroll );
         return false;
      }
   }*/

   return false;
}

void Continuation::invoke( const Item& retval )
{
   // passive phase
   m_bPhase = false;

   StackFrame* cframe = m_vm->currentFrame();
   m_stack.clear();
   m_stack.copyOnto( m_vm->stack(), m_stackLevel,
         m_vm->stackBase() - cframe->m_param_count - VM_FRAME_SPACE - m_stackLevel );

   // the PC will be in our return frame.
   m_tgtSymbol = cframe->m_symbol;
   m_tgtLModule = cframe->m_module;
   m_tgtPC = cframe->m_ret_pc;
   m_stackBase = cframe->m_stack_base;
   m_vm->regA() = retval;

   // Unroll the stack
   while( m_vm->stackBase() > m_stackLevel )
   {
      // neutralize post-processors
      m_vm->returnHandler( 0 );
      m_vm->callReturn();
      /*if ( m_vm->breakRequest() )
      {
         // exit from the C frame, but be sure to be called back
         m_vm->returnHandler( Continuation::unroll );
         return;
      }*/
   }
}




ContinuationCarrier::ContinuationCarrier( const CoreClass* cc ):
   CoreObject( cc ),
   m_cont(0),
   m_mark(0)
   {}

ContinuationCarrier::ContinuationCarrier( const ContinuationCarrier& other ):
   CoreObject( other ),
   m_mark(0)
{
   if( other.m_cont != 0 )
      m_cont = new Continuation( *other.m_cont );
   else
      m_cont = 0;
}

ContinuationCarrier::~ContinuationCarrier()
{
   delete m_cont;
}

ContinuationCarrier *ContinuationCarrier::clone() const
{
   return new ContinuationCarrier( *this );
}

bool ContinuationCarrier::setProperty( const String &prop, const Item &value )
{
   uint32 pos;
   if( m_generatedBy->properties().findKey( prop, pos ) )
      readOnlyError( prop );
   return false;
}

bool ContinuationCarrier::getProperty( const String &prop, Item &value ) const
{
   return defaultProperty( prop, value );
}

void ContinuationCarrier::gcMark( uint32 mark )
{
   if( mark == m_mark )
      return;

   m_mark = mark;
   memPool->markItem( m_citem );
   if ( m_cont != 0 )
   {
      m_cont->stack().gcMark( mark );
   }

}

CoreObject* ContinuationCarrier::factory( const CoreClass* cls, void* , bool  )
{
   return new ContinuationCarrier( cls );
}


}
