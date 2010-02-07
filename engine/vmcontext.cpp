/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/vm.h>
#include "vmsema.h"
#include <falcon/vmcontext.h>
#include <falcon/traits.h>
#include <falcon/genericvector.h>
#include <falcon/sys.h>

#define VM_STACK_MEMORY_THRESHOLD 128


namespace Falcon {

//==================================
// Deletor for the frame list.

VMContext::VMContext():
   m_frames(0),
   m_spareFrames(0)
{
   m_sleepingOn = 0;

   m_schedule = 0.0;
   m_priority = 0;

   m_atomicMode = false;

   m_tryFrame = 0;

   m_pc = 0;
   m_pc_next = 0;
   m_symbol = 0;
   m_lmodule = 0;

   m_frames = allocFrame();
}

VMContext::VMContext( const VMContext& other ):
   m_frames(0),
   m_spareFrames(0)
{
   m_sleepingOn = 0;

   m_schedule = 0.0;
   m_priority = 0;
   
   m_atomicMode = false;

   m_tryFrame = 0;

   m_pc = other.m_pc;
   m_pc_next = other.m_pc_next;
   m_symbol = other.m_symbol;
   m_lmodule = other.m_lmodule;

   m_frames = allocFrame();
   // reset stuff for the first frame,
   // as allocFrame doesn't clear everything for performance reasons.
   m_frames->m_param_count = 0;
}


VMContext::~VMContext()
{
   StackFrame* frame = m_spareFrames;
   while ( frame != 0 )
   {
      StackFrame* gone = frame;
      frame = frame->prev();
      delete gone;
   }

   frame = m_frames;
   while ( frame != 0 )
   {
      StackFrame* gone = frame;
      frame = frame->prev();
      delete gone;
   }

   m_frames = m_spareFrames = 0;
}


void VMContext::scheduleAfter( numeric secs )
{
   m_schedule = Sys::_seconds() + secs;
}


void VMContext::waitOn( VMSemaphore* sem, numeric secs )
{
   if( secs < 0.0 )
      m_schedule = -1.0;
   else
      m_schedule =  Sys::_seconds() + secs;

   m_sleepingOn = sem;
}

void VMContext::wakeup( bool signaled )
{
   if ( m_sleepingOn != 0 )  // overkill, but...
   {
      m_sleepingOn->unsubscribe( this );
      m_sleepingOn = 0;
      m_schedule = 0.0;  // immediately runnable

      // don't change the A status if not sleeping on a semaphore.
      regA().setBoolean(signaled); // we have not been awaken, and must return false
   }
}

void VMContext::signaled()
{
   if ( m_sleepingOn != 0 )  // overkill, but...
   {
      // Don't unsubscribe; the semaphore is unsubscribing us.
      m_sleepingOn = 0;
      m_schedule = 0.0;  // immediately runnable
   }

   regA().setBoolean(true); // we have not been awaken, and must return false}
}


StackFrame* VMContext::createFrame( uint32 paramCount, ext_func_frame_t frameEndFunc )
{
   StackFrame* frame = allocFrame();

   frame->prepareParams( m_frames, paramCount );

   frame->m_symbol = symbol();
   frame->m_module = lmodule();

   frame->m_ret_pc = pc_next();
   frame->m_call_pc = pc();
   frame->m_break = false;
   frame->m_prevTryFrame = m_tryFrame;

   frame->m_try_base = VMachine::i_noTryFrame;

   // parameter count.
   frame->m_param_count = paramCount;

   // iterative processing support
   frame->m_endFrameFunc = frameEndFunc;

   frame->m_self.setNil();
   frame->m_binding = regBind();

   return frame;
}


void VMContext::fillErrorTraceback( Error &error )
{
   fassert( ! error.hasTraceback() );

   const Symbol *csym = m_symbol;
   if ( csym != 0 )
   {
      uint32 curLine;
      if( csym->isFunction() )
      {
         curLine = csym->module()->getLineAt( csym->getFuncDef()->basePC() + pc() );
      }
      else {
         // should have been filled by raise
         curLine = error.line();
      }

      error.addTrace( csym->module()->name(), csym->name(),
         curLine,
         pc() );
   }

   StackFrame* frame = currentFrame();
   while( frame != 0 )
   {
      const Symbol *sym = frame->m_symbol;
      if ( sym != 0 )
      { // possible when VM has not been initiated from main
         uint32 line;
         if( sym->isFunction() )
            line = sym->module()->getLineAt( sym->getFuncDef()->basePC() + frame->m_call_pc );
         else
            line = 0;

         error.addTrace( sym->module()->name(), sym->name(), line, frame->m_call_pc );
      }

      frame = frame->prev();
   }
}


void VMContext::addFrame( StackFrame* frame )
{
   frame->prev( m_frames );
   m_frames = frame;
}

StackFrame* VMContext::popFrame()
{
   if ( m_frames == 0 )
      return 0;

   StackFrame *ret = m_frames;
   m_frames = m_frames->prev();
   ret->prev(0);
   return ret;
}

StackFrame* VMContext::allocFrame()
{
   if( m_spareFrames != 0 )
   {
      StackFrame* ret = m_spareFrames;
      m_spareFrames = m_spareFrames->prev();
      ret->resizeStack(0);

      ret->prev(0);
      ret->m_try_base = VMachine::i_noTryFrame;
      ret->m_prevTryFrame = 0;
      ret->m_module = 0;
      ret->m_symbol = 0;

      return ret;
   }

   return new StackFrame;
}


void VMContext::disposeFrame( StackFrame* frame )
{
   frame->prev( m_spareFrames );
   m_spareFrames = frame;
}

void VMContext::disposeFrames( StackFrame* first, StackFrame* last )
{
   last->prev( m_spareFrames );
   m_spareFrames = first;
}

void VMContext::resetFrames()
{
   if( m_frames == 0 )
   {
      m_frames = allocFrame();
   }
   else
   {
      StackFrame* top = m_frames->prev();
      m_frames->prev( 0 );
      m_frames->resizeStack(0);

      if ( top != 0 )
      {
         StackFrame* last = top;
         while ( last->prev() != 0 )
         {
            last = last->prev();
         }
         disposeFrames( top, last );
      }
   }

   /*
   if ( m_frames != 0 )
   {
      StackFrame* last = m_frames;
      while ( last->prev() != 0 )
      {
         last = last->prev();
      }
      disposeFrames( m_frames, last );
      m_frames = 0;
   }
   */


   m_frames->m_symbol = 0;
   m_frames->m_module = 0;
   m_tryFrame = 0;
}


void VMContext::pushTry( uint32 landingPC )
{
   Item frame1( (((int64) landingPC) << 32) | (int64) currentFrame()->m_try_base );
   m_tryFrame = currentFrame();
   currentFrame()->pushItem( frame1 );
   currentFrame()->m_try_base = currentFrame()->stackSize();
}

void VMContext::popTry( bool moveTo )
{
   // If the try frame is wrong or not in current stack frame...
   if( m_tryFrame == 0 )
   {
      throw new CodeError( ErrorParam( e_stackuf, __LINE__ ).
         origin( e_orig_vm ) );
   }

   // search the frame to pop
   while ( currentFrame() != m_tryFrame )
   {
      disposeFrame( popFrame() );
   }

   // get the frame and resize the stack
   currentFrame()->resizeStack( currentFrame()->m_try_base );
   fassert( currentFrame()->topItem().isInteger() );
   int64 tf_land = currentFrame()->topItem().asInteger();
   currentFrame()->pop(1);

   // Change the try frame, and eventually move the PC to the proper position
   currentFrame()->m_try_base = (uint32) tf_land;
   if( ((uint32) tf_land) == VMachine::i_noTryFrame )
   {
      m_tryFrame = currentFrame()->m_prevTryFrame;
   }

   if( moveTo )
   {
      pc_next() = (uint32)(tf_land>>32);
      pc() = pc_next();
   }
}


bool VMContext::callReturn()
{
   // Get the stack frame.
   StackFrame &frame = *currentFrame();

   // change symbol
   symbol( frame.m_symbol );
   pc_next() = frame.m_ret_pc;

   // eventually change active module.
   lmodule( frame.m_module );

   // reset try frame
   m_tryFrame = frame.m_prevTryFrame;

   regBind() = frame.m_binding;

   // reset stack base and resize the stack
   disposeFrame( popFrame() );
   // currentFrame may be zero, but in that case m_param_count would be zero too
   if ( frame.m_param_count != 0 )
   {
      fassert( currentFrame() != 0 );
      currentFrame()->pop( frame.m_param_count );
   }

   return frame.m_break;
}


}

/* end of vmcontext.cpp */
