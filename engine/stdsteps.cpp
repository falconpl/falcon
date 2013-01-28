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
#include <falcon/module.h>
namespace Falcon
{

void StdSteps::PStepDupliTop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->addSpace(1);
   ctx->opcodeParam(0) = ctx->opcodeParam(1);
}

void StdSteps::PStepDupliTop::describeTo( String& s, int ) const
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


void StdSteps::PStepDupliTop2::describeTo( String& s, int) const
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


void StdSteps::PStepDupliTop3::describeTo( String& s, int ) const
{
   s = "PStepDupliTop3";
}


void StdSteps::PStepSwapTop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(0).swap(ctx->opcodeParam(1));
}

void StdSteps::PStepSwapTop::describeTo( String& s, int ) const
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

void StdSteps::PStepSwapTopWith2::describeTo( String& s, int ) const
{
   s = "PStepSwapTopWith2";
}


void StdSteps::PStepCopyDown2::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(2) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown2::describeTo( String& s, int ) const
{
   s = "PStepCopyDown2";
}

void StdSteps::PStepCopyDown3::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(3) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown3::describeTo( String& s, int ) const
{
   s = "PStepCopyDown3";
}

void StdSteps::PStepCopyDown4::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(4) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown4::describeTo( String& s, int ) const
{
   s = "PStepCopyDown4";
}

void StdSteps::PStepCopyDown5::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(5) = ctx->opcodeParam(0);
}

void StdSteps::PStepCopyDown5::describeTo( String& s, int ) const
{
   s = "PStepCopyDown5";
}


void StdSteps::PStepPop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->popData();
}

void StdSteps::PStepPop::describeTo( String& s, int ) const
{
   s = "PStepPop";
}



void StdSteps::PStepPop2::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->popData(2);
}

void StdSteps::PStepPop2::describeTo( String& s, int ) const
{
   s = "PStepPop2";
}


void StdSteps::PStepPop3::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->popData(3);
}

void StdSteps::PStepPop3::describeTo( String& s, int ) const
{
   s = "PStepPop3";
}


void StdSteps::PStepDragDown::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->opcodeParam(1) = ctx->topData();
   ctx->popData();
}

void StdSteps::PStepDragDown::describeTo( String& s, int ) const
{
   s = "PStepDragDown";
}

void StdSteps::PStepAddSpace::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->addDataSlot();
}

void StdSteps::PStepAddSpace::describeTo( String& s, int ) const
{
   s = "PStepAddSpace";
}

void StdSteps::PStepPushNil::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->addDataSlot();
   ctx->topData().setNil();
}

void StdSteps::PStepPushNil::describeTo( String& s, int ) const
{
   s = "PStepPushNil";
}

void StdSteps::PStepReturnFrame::apply_( const PStep*, VMContext* ctx )
{
   ctx->returnFrame();
}

void StdSteps::PStepReturnFrame::describeTo( String& s, int ) const
{
   s = "PStepReturnFrame";
}


void StdSteps::PStepReturnFrameWithTop::apply_( const PStep*, VMContext* ctx )
{
   ctx->returnFrame( ctx->topData() );
}

void StdSteps::PStepReturnFrameWithTop::describeTo( String& s, int ) const
{
   s = "PStepReturnFrameWithTop";
}

void StdSteps::PStepReturnFrameWithTopDoubt::apply_( const PStep*, VMContext* ctx )
{
   ctx->returnFrameND( ctx->topData() );
}

void StdSteps::PStepReturnFrameWithTopDoubt::describeTo( String& s, int ) const
{
   s = "PStepReturnFrameWithTopDoubt";
}


void StdSteps::PStepReturnFrameWithTopEval::apply_( const PStep*, VMContext* ctx )
{
   ctx->returnFrameEval( ctx->topData() );
}

void StdSteps::PStepReturnFrameWithTopEval::describeTo( String& s, int ) const
{
   s = "PStepReturnFrameWithTopEval";
}

void StdSteps::PStepReturnFrameWithTopDoubtEval::apply_( const PStep*, VMContext* ctx )
{
   ctx->returnFrameNDEval( ctx->topData() );
}

void StdSteps::PStepReturnFrameWithTopDoubtEval::describeTo( String& s, int ) const
{
   s = "PStepReturnFrameWithTopDoubtEval";
}


void StdSteps::PStepLocalFrame::apply_( const PStep*, VMContext* ctx )
{
   register int base = ctx->currentCode().m_seqId;
   ctx->popCode();
   // 0 is marker for unused. The real base is seqId - 1.
   if( base > 0 ) {
      Item top = ctx->topData();
      ctx->unrollLocalFrame( base-1 );
      ctx->topData() = top;
   }
}
void StdSteps::PStepLocalFrame::describeTo( String& s, int ) const
{
   s = "PStepLocalFrame";
}


void StdSteps::PStepLocalFrameExec::apply_( const PStep*, VMContext* ctx )
{
   register int base = ctx->currentCode().m_seqId;
   ctx->popCode();
   // 0 is marker for unused. The real base is seqId - 1.
   if( base > 0 ) {
      Item top = ctx->topData();
      ctx->unrollLocalFrame( base-1 );
      ctx->topData() = top;
   }
   ctx->callItem(ctx->topData());
}
void StdSteps::PStepLocalFrameExec::describeTo( String& s, int ) const
{
   s = "PStepLocalFrameExec";
}

void StdSteps::PStepUnrollToLoop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->unrollToLoopBase();
}

void StdSteps::PStepUnrollToLoop::describeTo( String& s, int ) const
{
   s = "PStepUnrollToLoop";
}


void StdSteps::PStepUnrollToNext::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->unrollToNextBase();
}

void StdSteps::PStepUnrollToNext::describeTo( String& s, int ) const
{
   s = "PStepUnrollToNext";
}


void StdSteps::PStepRaiseTop::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   ctx->raiseItem( ctx->topData() );
}

void StdSteps::PStepRaiseTop::describeTo( String& s, int ) const
{
   s = "PStepRaiseTop";
}

void StdSteps::PStepFillInstance::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();
   Item& inst = ctx->topData();
   Class* cls = 0;
   void* data = 0;

   if( inst.asClassInst(cls, data) )
   {
      // is this a valid class instance?
      if ( cls->name().size() > 0 && cls->name().getCharAt(0) == '%' )
      {
         Module* owner = cls->module();
         if( owner )
         {
            String instName = cls->name().subString(1);
            Item* gval = owner->getGlobalValue(instName);
            if( gval != 0 )
            {
               // Success!
               gval->assignFromLocal(inst);
               ctx->popData();
               return;
            }

            // failed to find the variable
            throw ctx->runtimeError( e_no_cls_inst, instName + " inst of " + cls->name() );
         }
      }
      else {
            throw new CodeError(
                     ErrorParam( e_no_cls_inst, __LINE__, SRC )
                     .origin( ErrorParam::e_orig_vm )
                     .symbol(cls->name())
                     .extra("not instanceable")
                     );
      }
   }
   else {
      throw new CodeError(
               ErrorParam( e_no_cls_inst, __LINE__, SRC )
                           .origin( ErrorParam::e_orig_vm )
                           .extra("Not a class")
                           );
   }


}

void StdSteps::PStepFillInstance::describeTo( String& s, int ) const
{
   s = "PStepFillInstance";
}



void StdSteps::PStepFillAttribute::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();

   Item& value = ctx->topData();
   Item& iattrib = ctx->opcodeParam(1);
   Attribute* op = static_cast<Attribute*>(iattrib.asOpaque());
   op->value().copyFromLocal(value);

   ctx->popData(2);
}

void StdSteps::PStepFillAttribute::describeTo( String& s, int ) const
{
   s = "PStepFillAttribute";
}


}

/* end of stdsteps.cpp */
