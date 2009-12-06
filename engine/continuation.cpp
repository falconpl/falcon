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
   m_bComplete( false )
{
   m_context = vm->currentContext();
}


Continuation::Continuation( const Continuation& e ):
   m_vm( e.m_vm ),
   m_stack( e.m_stack ),
   m_tgtSymbol( e.m_tgtSymbol ),
   m_tgtLModule( e.m_tgtLModule ),
   m_bComplete( e.m_bComplete )
{
   m_context = e.m_context;
   m_stackLevel = e.m_stackLevel;
   m_tgtPC = e.m_tgtPC;

}


Continuation::~Continuation()
{

}


bool Continuation::jump()
{
   if ( m_vm->currentContext()->atomicMode() )
   {
      throw new CodeError( ErrorParam( e_cont_atomic, __LINE__ )
            .origin( e_orig_vm ) );
   }

   m_stackLevel = m_vm->stackBase();

   if ( m_tgtSymbol != 0 )
   {
      fassert( m_tgtSymbol->isFunction() )
      if( m_bComplete )
      {
         throw new CodeError( ErrorParam( e_cont_out, __LINE__ )
               .origin( e_orig_vm )  );
      }
      m_bComplete = true;

      // remove our frame, or we'll be called twice.
      m_vm->stack().copyOnto( m_stackLevel, m_stack, 0, m_stack.length() );
      m_vm->currentContext()->symbol( m_tgtSymbol );
      m_vm->currentContext()->lmodule( m_tgtLModule );
      m_vm->currentContext()->pc_next() = m_tgtPC;
      m_vm->currentContext()->stackBase() = m_stackBase;
      return true;
   }

   m_bComplete = true;
   return false;
}



void Continuation::suspend( const Item& retval )
{
   if ( m_vm->currentContext()->atomicMode() )
   {
      throw new CodeError( ErrorParam( e_cont_atomic, __LINE__ )
            .origin( e_orig_vm ) );
   }

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

   // for sure, we need more call
   m_bComplete = false;

   // Unroll the stack
   while( m_vm->stackBase() >= m_stackLevel )
   {
      // neutralize post-processors
      m_vm->returnHandler( 0 );
      m_vm->callReturn();
      fassert( ! m_vm->breakRequest() );
   }
}

//=============================================================
// Continuation Carrier


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
