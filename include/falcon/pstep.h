/*
   FALCON - The Falcon Programming Language.
   FILE: pstep.h

   Common interface to VM processing step.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 18:01:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_PSTEP_H
#define FALCON_PSTEP_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/sourceref.h>

namespace Falcon {

class VMContext;

/** Common interface to VM processing step.
 *
 * This is the common base class for statements, expressions and statement sequences,
 * which are the basic components of the Virtual Machine code processing.
 *
 * These are the 3 basic elements that can be processed by the virtual machine.
 * The VM scans its code stack, calling the apply() method of the topmost element.
 * When a sequence "apply" is invoked, it starts calling the perform() method of all its
 * members; they can be performed on the spot, acting immediately on the virtual machine,
 * or push themselves in the code stack and return the control to the calling VM.
 *
 * Expressions can be also processed by statements that can call their eval() member directly.
 */
class FALCON_DYN_CLASS PStep
{
public:
   /** Size of each depth scale in describe(). */
   static const int depthIndent = 2;

   inline PStep( int line=0, int chr=0 ):
      m_bIsLoopBase(false),
      m_bIsNextBase(false),
      m_bIsComposed(false),
      m_catchMode(false),
      m_sr(line, chr)
   {}

   inline virtual ~PStep() {}

   /** Convert into a string.
    \param target A target string where to write the pstep representation.

    The default base class function does nothing. This is useful for
    pstep that are not part of the syntactic tree, but just of the
    VM code.
    */
   virtual void describeTo( String& target ) const;

   inline String describe() const {
      String temp; describeTo(temp); return temp;
   }

   /** Sparsely used function marking steps with special significance.

    */
   inline virtual uint32 flags() const { return 0; }

   /** Apply function.
    \param self The PStep that was applied here.
    \param ctx The virtual machine context where this apply function was run

    This is the callback function that the virtual machine will invoke when
    this step is found in the code stack. As such, the step being called via
    the apply function is \b granted to be on top of the code stack of the
    \b ctx it receives.

    \note the apply function is called exclusively by a Virtual Machine or
    Virtual Machine Context related operation on the top item in the code
    stack.

    The function could be called also by a composite PStep. For instance,
    it can be called by a parent expression or by a SynTree, which is a
    collection of special subclasses of PStep (more specifically, a collection
    of Statement instances).

    The composite PStep will check if the called sub-steps have changed the
    code stack, and in case they did, it will yield the control to its caller.

    The pattern for composite step is as follows:
    @code
    // I am at top of VMContext
    while not done
       get next substep
       call next substep apply by pushing it in the context
       Am I still at top of the VMcontext? -- if not, return
    end
    pop myself from VMContext.
    @endcocde

    There are two ways to check if the VMContext has been changed. It is possible
    to check if the topmost code frame is changed,
    @code
    CodeFrame& topCode = ctx->currentCode(); // yep, that's me
    doSomething();
    if( &topCode != &ctx->currentCode() )
       return;
    ctx->popCode(); // remove me.
    @endcode

    Although fast, this method is subject to false positive in case the code
    in doSomething pushed something in the stack but then removed it, as the
    stack might have been invalidated in the meanwhile.

    However, this is hardly a problem as the stack soon stabilizes to a size
    accomodating the program depth, and the code must be designed to properly
    be resumed after returning. However, if this causes some problem, or if
    arranging for return is slower than peforming a little more complex check,
    this second pattern can be used:

    @code
    long depth = ctx->codeSize();
    doSomething();
    if( ctx->wentDeepSized( depth ) )
       return;
    @endcode

    The check is slower (this requires some pointer math) but avoids
    false postives.
    */
   typedef void (*apply_func)(const PStep* self, VMContext* ctx);

   apply_func apply;

   const SourceRef& sr() const { return m_sr; }

   /** Sets or changes the declaration position.
    \param l line at which this PStep was created.
    \param c character at which this PStep was created.
    */
   PStep& decl( int line, int chr ) {
      m_sr.line(line);
      m_sr.chr(chr);
      return *this;
   }

   /** Returns the line where this PStep was declared in source.
    If the PStep wasn't generated from source, the line will be 0.
    */
   inline int line() const { return m_sr.line(); }

   /** Returns the character where this PStep was declared in source.
    If the PStep wasn't generated from source, the character will be 0.
   */
   inline int chr() const { return m_sr.chr(); }

   inline bool isComposed() const { return m_bIsComposed; }
   inline bool isNextBase() const { return m_bIsNextBase; }
   inline bool isLoopBase() const { return m_bIsLoopBase; }
   inline bool isTry() const { return m_catchMode != 0; }
   inline bool isTracedCatch() const { return m_catchMode == 2; }

   inline void setNextBase() { m_bIsNextBase = true; }
   /** Declare that this pstep is a try-catch statement.
    *
    *  The pstep must be derived from StmtTry
    *
    *  Remember to push via VMContext::pushWithUnrollContext
    */
   inline void setTry() { m_catchMode = 1; }
   /** Declared this as a traced catch.
    *
    *  If this pstep is a catch clause being part of a try-catch statement,
    *  then this option forces the VM to generate a traceback immediately
    *  as the catch path is followed.
    *
    *  If this pstep is not part of try-catch statement, then it is
    *  considered a SynTree that will act as a catch-all clause
    *  for any raised error. In this case,
    *  remember to push via VMContext::pushWithUnrollContext
    */
   inline void setTracedCatch() { m_catchMode = 2; }

   /**
    * Returns a pre-rendered prefix for rendering on output.
    */
   static String renderPrefix(int32 size);

   static int32 relativeDepth(int32 depth)
   {
      return depth < 0 ? depth : -depth-1;
   }

protected:
   bool m_bIsLoopBase;
   bool m_bIsNextBase;
   bool m_bIsComposed;
   byte m_catchMode;

private:

   SourceRef m_sr;
};

}


#define FALCON_DECLARE_INTERNAL_PSTEP( name ) \
      class FALCON_DYN_CLASS PStep ## name : public PStep\
      {\
      public:\
         PStep ## name(){apply = apply_;}\
         virtual ~PStep ## name() {}\
         static void apply_( const PStep*, VMContext* ctx );\
         virtual void describeTo( String& desc ) const\
         {\
            desc = #name;\
         }\
      };\
      PStep ## name m_step ## name ;


      //Need to do something about this (m_Owner)
#define FALCON_DECLARE_INTERNAL_PSTEP_OWNED( name, ownerclass ) \
      class FALCON_DYN_CLASS PStep ## name : public PStep\
      {\
      public:\
         PStep ## name( ownerclass* owner): m_owner(owner) {apply = apply_;}\
         virtual ~ PStep ## name() {}\
         static void apply_( const PStep*, VMContext* ctx );\
         virtual void describeTo( String& desc ) const\
         {\
            desc = #ownerclass "::" #name;\
         }\
         ownerclass* m_owner;\
      };\
      PStep ## name m_step ## name ;

#define FALCON_DEFINE_INTERNAL_PSTEP( Class__, Name__ )\
         void Class__::PStep ## Name__ ::apply_

#define FALCON_DEFINE_INTERNAL_PSTEP_P1( Class__, Name__ )\
         void Class__::PStep ## Name__ ::apply_( const PStep* , VMContext* ctx )

#define FALCON_DEFINE_INTERNAL_PSTEP_P2( Class__, Name__ )\
         void Class__::PStep ## Name__ ::apply_( const PStep* pstep, VMContext* ctx )

#endif

/* end of pstep.h */
