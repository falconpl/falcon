/*
   FALCON - The Falcon Programming Language
   FILE: stackframe.cpp

   Implementation for stack frame functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 15 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation for stack frame functions
*/

#include <falcon/stackframe.h>
#include <falcon/mempool.h>

namespace Falcon
{

StackFrame::StackFrame( const StackFrame& other ):
   m_break( other.m_break ),
   m_ret_pc( other.m_ret_pc ),
   m_call_pc( other.m_call_pc ),
   m_param_count( other.m_param_count ),
   m_try_base( other.m_try_base ),
   m_symbol( other.m_symbol ),
   m_module( other.m_module ),
   m_endFrameFunc( other.m_endFrameFunc ),
   m_prevTryFrame( other.m_prevTryFrame ),
   m_self( other.m_self ),
   m_binding( other.m_binding ),
   m_params( other.m_params ),
   m_prev(0),
   m_stack(other.m_stack)
{

}

StackFrame* StackFrame::copyDeep( StackFrame** bottom )
{
   StackFrame* curTryFrame = 0;

   StackFrame* top = 0;
   StackFrame* current = 0;
   StackFrame* orig = this;

   // Copy this frame.
   while( orig != 0 )
   {
      StackFrame* nframe = new StackFrame(*orig);
      // is this the top frame?
      if ( top == 0 )
      {
         top = nframe;
         current = nframe;
      }
      else
      {
         current->prev( nframe );
         current->prepareParams( nframe, current->m_param_count );
      }

      // Did we reach a given try frame ?
      if( orig == curTryFrame )
      {
         // then change all the matching fames with this one
         StackFrame* review = top;
         while( review != nframe )
         {
            if( review->m_prevTryFrame == orig )
               review->m_prevTryFrame = nframe;
            review = review->prev();
         }
      }

      current = nframe;

      curTryFrame = orig->m_prevTryFrame;
      orig = orig->prev();
   }

   if ( bottom != 0 )
      *bottom = current;
   return top;
}

void StackFrame::gcMark( uint32 mark )
{
   uint32 sl = stackSize();
   memPool->markItem( m_self );
   memPool->markItem( m_binding );

   for( uint32 pos = 0; pos < sl; pos++ ) {
      memPool->markItem( stackItems()[ pos ] );
   }
}

}


/* end of stackframe.cpp */
