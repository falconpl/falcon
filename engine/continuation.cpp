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
   m_callingFrame(0),
   m_top(0),
   m_bottom(0),
   m_bComplete( false )
{
   m_context = vm->currentContext();
}


Continuation::Continuation( const Continuation& e ):
   m_vm( e.m_vm ),
   m_tgtSymbol( e.m_tgtSymbol ),
   m_tgtLModule( e.m_tgtLModule ),
   m_bComplete( e.m_bComplete )
{
   if( e.m_top != 0 )
   {
      m_top = e.m_top->copyDeep( &m_bottom );
   }
   else
   {
      m_top = 0;
      m_bottom = 0;
   }

   m_callingFrame = e.m_callingFrame;
   m_context = e.m_context;
   m_stackLevel = e.m_stackLevel;
   m_tgtPC = e.m_tgtPC;
}


Continuation::~Continuation()
{
   StackFrame* frame = m_top;
   while( frame != 0 )
   {
      StackFrame* f = frame;
      frame = frame->prev();
      delete f;
   }
}


bool Continuation::jump()
{
   if ( m_vm->currentContext()->atomicMode() )
   {
      throw new CodeError( ErrorParam( e_cont_atomic, __LINE__ )
            .origin( e_orig_vm ) );
   }

   m_callingFrame = m_vm->currentFrame();

   if ( m_tgtSymbol != 0 )
   {
      fassert( m_tgtSymbol->isFunction() || m_tgtSymbol->isExtFunc() );
      if( m_bComplete )
      {
         throw new CodeError( ErrorParam( e_cont_out, __LINE__ )
               .origin( e_orig_vm )  );
      }
      m_bComplete = true;

      // engage the previous frame
      m_bottom->prev( m_callingFrame );
      for ( uint32 i = 0; i < m_params.length(); ++i )
         m_callingFrame->stack().append( m_params[i] );
      m_bottom->prepareParams( m_callingFrame, m_bottom->m_param_count );

      // Set the new frame
      m_context->setFrames( m_top );
      m_bottom = m_top = 0;

      // jump
      m_vm->currentContext()->symbol( m_tgtSymbol );
      m_vm->currentContext()->lmodule( m_tgtLModule );
      m_vm->currentContext()->pc_next() = m_tgtPC;
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

   // find the calling frame.
   StackFrame* frame = m_vm->currentFrame();
   while( frame->prev() != m_callingFrame )
   {
      frame = frame->prev();
   }

   // save the original parameters
   m_params.clear();
   for( uint32 i = 0; i < frame->m_param_count; i++ )
   {
      m_params.append( frame->m_params[i] );
   }

   // disengage the stack.
   frame->prev(0);
   m_bottom = frame;
   m_top = m_vm->currentFrame();
   // and remove the parameters
   m_callingFrame->pop( frame->m_param_count );
   m_context->setFrames( m_callingFrame );

   // the PC will be in our return frame.
   m_tgtSymbol = m_callingFrame->m_symbol;
   m_tgtLModule = m_callingFrame->m_module;
   m_tgtPC = m_callingFrame->m_ret_pc;
   m_vm->regA() = retval;

   // for sure, we need more call
   m_bComplete = false;
}


bool Continuation::updateSuspendItem( const Item& itm )
{
   if ( m_top == 0 )
      return false;

   StackFrame* t = m_top;
   while( t->prev() != m_bottom )
   {
      t = t->prev();
   }

   if( t->m_param_count )
      t->m_params[0] = itm;

   return true;
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
   m_citem( other.m_citem ),
   m_mark(0)
{
   if( other.m_cont != 0 )
      m_cont = new Continuation( *other.m_cont );
   else
      m_cont = 0;

   getMethod("_suspend",  suspendItem() );

   m_cont->updateSuspendItem( suspendItem() );
}

ContinuationCarrier::~ContinuationCarrier()
{
   delete m_cont;
}

ContinuationCarrier *ContinuationCarrier::clone() const
{
   // for now, uncloneable
   return 0;

   //return new ContinuationCarrier( *this );
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
      m_cont->params().gcMark( mark );
      StackFrame* sf = m_cont->frames();
      while( sf != 0 )
      {
         sf->gcMark( mark );
         sf = sf->prev();
      }
   }

}

CoreObject* ContinuationCarrier::factory( const CoreClass* cls, void* , bool  )
{
   return new ContinuationCarrier( cls );
}


}
