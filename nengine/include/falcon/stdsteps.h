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

#ifndef _FALCON_STDSTEPS_H
#define _FALCON_STDSTEPS_H

#include <falcon/setup.h>
#include <falcon/pstep.h>

namespace Falcon 
{

class VMContext;

/** Collection of commonly used standard PSteps.
 
 This class holds some common use PSteps that can be freely pushed
 in any VM.
 
 This obviate the needs to locally create and destroy this PSteps, or
 to create Private PSteps doing repetitive work.
 
 The psteps ending with _ can be added to a PCode, as they don't remove
 themselves; the others can be added directly to the virtual machine code
 stack.
 */
class StdSteps
{
public:
   
   //======================================================
   // Expression versions.
   //
   
   /** Duplicate the topmost item in the stack. */
   class PStepDupliTop: public PStep
   {
   public:
      PStepDupliTop() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDupliTop m_dupliTop;
   
   /** Duplicate the topmost 2 items in the stack.  */
   class PStepDupliTop2: public PStep
   {
   public:
      PStepDupliTop2() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDupliTop2 m_dupliTop2;
   
   /** Duplicate the topmost 3 items in the stack.  */
   class PStepDupliTop3: public PStep
   {
   public:
      PStepDupliTop3() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDupliTop3 m_dupliTop3;

   
   /** Swap the topmost item and the second-topmost one in the stack. */
   class PStepSwapTop: public PStep
   {
   public:
      PStepSwapTop() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   
   PStepSwapTop m_swapTop;
   
   /** Swap the topmost item putting it below the topmost 2. 
    
    This is used when having evaluated an expression that must be sent to
    an array setIndex, as setIndex wants \b value, \b array, \b index
    in the stack in this order.
    */   
   class PStepSwapTopWith2: public PStep
   {
   public:
      PStepSwapTopWith2() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   
   PStepSwapTopWith2 m_swapTopWith2;
   
   /** Base class for copy-down steps. 
    Copy-down steps copy the topmost item to the nth item in the stack.
    
    \note the VM model only ensures that the items in the current call frame
    are accessible randomly in the data stack, but no check is performed here.
    
    */
   class PStepCopyDown2: public PStep
   {
   public:
      PStepCopyDown2() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown2 m_copyDown2;
   
   class PStepCopyDown3: public PStep
   {
   public:
      PStepCopyDown3() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown3 m_copyDown3;
   
   class PStepCopyDown4: public PStep
   {
   public:
      PStepCopyDown4() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown4 m_copyDown4;
   
   class PStepCopyDown5: public PStep
   {
   public:
      PStepCopyDown5() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown5 m_copyDown5;

   /** Cleans the topmost value */
   class PStepPop: public PStep
   {
   public:
      PStepPop() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepPop m_pop;
   
   /** Pops the topmost value of the stack and copies it on the previous item. */
   class PStepDragDown: public PStep
   {
   public:
      PStepDragDown() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDragDown m_dragDown;

   /** Adds one empty space in top of the stack.
    The space added is nilled for safety.
    */
   class PStepAddSpace: public PStep
   {
   public:
      PStepAddSpace() {apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepAddSpace m_addSpace;
   
   //======================================================
   // PCode version
   //
   
   /** Duplicate the topmost item in the stack. */
   class PStepDupliTop_: public PStep
   {
   public:
      PStepDupliTop_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDupliTop_ m_dupliTop_;
   
   /** Duplicate the topmost 2 items in the stack.  */
   class PStepDupliTop2_: public PStep
   {
   public:
      PStepDupliTop2_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDupliTop2_ m_dupliTop2_;
   
   /** Duplicate the topmost 3 items in the stack.  */
   class PStepDupliTop3_: public PStep
   {
   public:
      PStepDupliTop3_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDupliTop3_ m_dupliTop3_;

   
   /** Swap the topmost item and the second-topmost one in the stack. */
   class PStepSwapTop_: public PStep
   {
   public:
      PStepSwapTop_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   
   PStepSwapTop_ m_swapTop_;
   
   /** Swap the topmost item putting it below the topmost 2. 
    
    This is used when having evaluated an expression that must be sent to
    an array setIndex, as setIndex wants \b value, \b array, \b index
    in the stack in this order.
    */   
   class PStepSwapTopWith2_: public PStep
   {
   public:
      PStepSwapTopWith2_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   
   PStepSwapTopWith2_ m_swapTopWith2_;
   
   /** Base class for copy-down steps. 
    Copy-down steps copy the topmost item to the nth item in the stack.
    
    \note the VM model only ensures that the items in the current call frame
    are accessible randomly in the data stack, but no check is performed here.
    
    */
   class PStepCopyDown2_: public PStep
   {
   public:
      PStepCopyDown2_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown2_ m_copyDown2_;
   
   class PStepCopyDown3_: public PStep
   {
   public:
      PStepCopyDown3_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown3_ m_copyDown3_;
   
   class PStepCopyDown4_: public PStep
   {
   public:
      PStepCopyDown4_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown4_ m_copyDown4_;
   
   class PStepCopyDown5_: public PStep
   {
   public:
      PStepCopyDown5_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepCopyDown5_ m_copyDown5_;

   /** Cleans the topmost value */
   class PStepPop_: public PStep
   {
   public:
      PStepPop_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepPop_ m_pop_;
   
   /** Pops the topmost value of the stack and copies it on the previous item. */
   class PStepDragDown_: public PStep
   {
   public:
      PStepDragDown_() { apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepDragDown_ m_dragDown_;
   
   class PStepAddSpace_: public PStep
   {
   public:
      PStepAddSpace_() {apply = apply_; }
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& ) const;
   };
   PStepAddSpace_ m_addSpace_;
};

}

#endif	/* _FALCON_STDSTEPS_H */

/* end of stdsteps.h */
