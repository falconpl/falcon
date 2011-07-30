/*
   FALCON - The Falcon Programming Language.
   FILE: stdsteps.h

   Standard misc PSteps commonly used in the virtual machine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 00:39:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/stdsteps.cpp"

#include <falcon/stdsteps.h>
#include <falcon/vmcontext.h>
namespace Falcon 
{

void StdSteps::PStepDupliTop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->addSpace(1);
   ctx->opcodeParam(0) = ctx->opcodeParam(1);
}

void StdSteps::PStepDupliTop::describe( String& s ) const
{
   s = "PStepDupliTop";
}


void StdSteps::PStepDupliTop2::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->addSpace(2);
   ctx->opcodeParam(1) = ctx->opcodeParam(3);
   ctx->opcodeParam(0) = ctx->opcodeParam(2);
}


void StdSteps::PStepDupliTop2::describe( String& s) const
{
   s = "PStepDupliTop2";
}


void StdSteps::PStepDupliTop3::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->addSpace(3);
   ctx->opcodeParam(2) = ctx->opcodeParam(5);
   ctx->opcodeParam(1) = ctx->opcodeParam(4);
   ctx->opcodeParam(0) = ctx->opcodeParam(3);
}


void StdSteps::PStepDupliTop3::describe( String& s ) const
{
   s = "PStepDupliTop3";
}


void StdSteps::PStepSwapTop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(0).swap(ctx->opcodeParam(1));
}

void StdSteps::PStepSwapTop::describe( String& s ) const
{
   s = "PStepSwapTop";
}

  

void StdSteps::PStepSwapTopWith2::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   Item top = ctx->topData();
   ctx->opcodeParam(0) = ctx->opcodeParam(1);
   ctx->opcodeParam(1) = ctx->opcodeParam(2);
   ctx->opcodeParam(2) = top;
}

void StdSteps::PStepSwapTopWith2::describe( String& s ) const
{
   s = "PStepSwapTopWith2";
}


void StdSteps::PStepCopyDown2::apply_( const PStep*, VMContext* ctx )
{   
   ctx->popCode();
   ctx->opcodeParam(2) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown2::describe( String& s ) const
{
   s = "PStepCopyDown2";
}

void StdSteps::PStepCopyDown3::apply_( const PStep*, VMContext* ctx )
{   
   ctx->popCode();
   ctx->opcodeParam(3) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown3::describe( String& s ) const
{
   s = "PStepCopyDown3";
}

void StdSteps::PStepCopyDown4::apply_( const PStep*, VMContext* ctx )
{   
   ctx->popCode();
   ctx->opcodeParam(4) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown4::describe( String& s ) const
{
   s = "PStepCopyDown4";
}

void StdSteps::PStepCopyDown5::apply_( const PStep*, VMContext* ctx )
{   
   ctx->popCode();
   ctx->opcodeParam(5) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown5::describe( String& s ) const
{
   s = "PStepCopyDown5";
}


void StdSteps::PStepPop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->popData();
}

void StdSteps::PStepPop::describe( String& s ) const
{
   s = "PStepPop";
}


void StdSteps::PStepDragDown::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(1) = ctx->topData();
   ctx->popData();
}

void StdSteps::PStepDragDown::describe( String& s ) const
{
   s = "PStepDragDown";
}

void StdSteps::PStepAddSpace::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->addDataSlot();
}

void StdSteps::PStepAddSpace::describe( String& s ) const
{
   s = "PStepAddSpace";
}

//======================================================
// PCode version
//
void StdSteps::PStepDupliTop_::apply_( const PStep*, VMContext* ctx )
{
   ctx->addSpace(1);
   ctx->opcodeParam(0) = ctx->opcodeParam(1);
}

void StdSteps::PStepDupliTop_::describe( String& s ) const
{
   s = "PStepDupliTop_";
}


void StdSteps::PStepDupliTop2_::apply_( const PStep*, VMContext* ctx )
{
   ctx->addSpace(2);
   ctx->opcodeParam(1) = ctx->opcodeParam(3);
   ctx->opcodeParam(0) = ctx->opcodeParam(2);
}


void StdSteps::PStepDupliTop2_::describe( String& s) const
{
   s = "PStepDupliTop2_";
}


void StdSteps::PStepDupliTop3_::apply_( const PStep*, VMContext* ctx )
{
   ctx->addSpace(3);
   ctx->opcodeParam(2) = ctx->opcodeParam(5);
   ctx->opcodeParam(1) = ctx->opcodeParam(4);
   ctx->opcodeParam(0) = ctx->opcodeParam(3);
}


void StdSteps::PStepDupliTop3_::describe( String& s ) const
{
   s = "PStepDupliTop3_";
}


void StdSteps::PStepSwapTop_::apply_( const PStep*, VMContext* ctx )
{
   ctx->opcodeParam(0).swap(ctx->opcodeParam(1));
}

void StdSteps::PStepSwapTop_::describe( String& s ) const
{
   s = "PStepSwapTop_";
}

  

void StdSteps::PStepSwapTopWith2_::apply_( const PStep*, VMContext* ctx )
{
   Item top = ctx->topData();
   ctx->opcodeParam(0) = ctx->opcodeParam(1);
   ctx->opcodeParam(1) = ctx->opcodeParam(2);
   ctx->opcodeParam(2) = top;
}

void StdSteps::PStepSwapTopWith2_::describe( String& s ) const
{
   s = "PStepSwapTopWith2_";
}


void StdSteps::PStepCopyDown2_::apply_( const PStep*, VMContext* ctx )
{   
   ctx->opcodeParam(2) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown2_::describe( String& s ) const
{
   s = "PStepCopyDown2_";
}

void StdSteps::PStepCopyDown3_::apply_( const PStep*, VMContext* ctx )
{   
   ctx->opcodeParam(3) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown3_::describe( String& s ) const
{
   s = "PStepCopyDown3_";
}

void StdSteps::PStepCopyDown4_::apply_( const PStep*, VMContext* ctx )
{   
   ctx->opcodeParam(4) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown4_::describe( String& s ) const
{
   s = "PStepCopyDown4_";
}

void StdSteps::PStepCopyDown5_::apply_( const PStep*, VMContext* ctx )
{   
   ctx->opcodeParam(5) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown5_::describe( String& s ) const
{
   s = "PStepCopyDown5_";
}


void StdSteps::PStepPop_::apply_( const PStep*, VMContext* ctx )
{
   ctx->popData();
}

void StdSteps::PStepPop_::describe( String& s ) const
{
   s = "PStepPop_";
}


void StdSteps::PStepDragDown_::apply_( const PStep*, VMContext* ctx )
{
   ctx->opcodeParam(1) = ctx->topData();
   ctx->popData();
}

void StdSteps::PStepDragDown_::describe( String& s ) const
{
   s = "PStepDragDown_";
}


void StdSteps::PStepAddSpace_::apply_( const PStep*, VMContext* ctx )
{
   ctx->addDataSlot();
}

void StdSteps::PStepAddSpace_::describe( String& s ) const
{
   s = "PStepAddSpace_";
}

}

/* end of stdsteps.cpp */
