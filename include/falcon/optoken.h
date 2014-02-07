/*
   FALCON - The Falcon Programming Language.
   FILE: optoken.h

   A token to simplify operand enter/exit in core classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 14:22:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_OPTOKEN_H
#define	_FALCON_OPTOKEN_H

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/vmcontext.h>

#include "vm.h"

namespace Falcon {

/** A token to simplify operand enter/exit in core classes.

   This class can be used by Class operand implementors to simplify
   the implementation.

 Througt the OpToken class, it's possible to produce simpler and safer code
 when writing Class operators that need to enter directly on the VM stack to
 access their paramenters.

 This model allows a good balance between performance, code safety and elegance.

 In practice, the OpToken class takes care to extract the operand parameters
 from the VM stack and clean it when the function exits, placing the operand
 result in the correct position of the VM stack.

 The stack unroll is not forced; it needs to be optional because operands may
 go deep in the virtual machine, asking for execution of a second step; also,
 in case of error the stack would be automatically unrolled at try position,
 so unrolling it in the OpToken would be a waste of time in the error handling.

 For this reasons, the exit() method must be called before returning the control
 to the VM; the abandon() method must be called if the token is going deep.

 When popping the stack from a second step, the stack should be popped manually
 from the current context, or the step may use the stepExit static method
 of this class as a semantic marker of the end of the end of the operand frame.

 In debug builds, the class is very verbose in checking correct usage semantic
 and prevent underflow errors. In release builds, all the checks are abandoned
 in favor of perfomance.

 The class has two sets of constructors, one using a VMContext and the other
 using a VMachine. If the caller has a reason to store the VMContext locally
 in the operator code, then using the VMContext based constructor will be
 faster, else the VMachine based constructor will extract the current context
 on their own.

 An example usage:
 @code
 virtual void MyClasss::op_something( VMContext* ctx, void* self )
 {
    // get the two operands of this class.
    Item *op1, op2;
    OpToken token( op1, op2 );

    ...
    // we're done
    token.exit( FALCON_GC_HANDLE(new String("The returned value")) );
 }
 @endcode

 \note The class is completely inlined and relatively efficient. However,
 if you want to use register class pointers to store operands for faster
 access, you should access directly the stack in the current VMContext.
 
 \see Class
 \see VMachine::operands
 */
class OpToken {
public:
   VMContext* m_vmc;
   int m_ops;

#ifndef NDEBUG
   bool m_bExited;
#endif

   /** Open an operator contet with a single operand.
    \param vmc The context of the virtual machine.
    \param op1 Where to store the first operand.
    */
   inline OpToken( VMContext* vmc, Item*& op1 ):
      m_vmc( vmc ),
      m_ops(0)
#ifndef NDEBUG
      , m_bExited(false)
#endif
   {
      op1 = &vmc->topData();
   }

   /** Open an operator contet with a single operand.
    \param vmc The context of the virtual machine.
    \param op1 Where to store the first operand (top-data).
    \param op2 Where to store the second operand (second-last).

    */
   inline OpToken( VMContext* vmc, Item*& op1, Item*& op2 ):
      m_vmc( vmc ),
      m_ops( 1 )
#ifndef NDEBUG
      , m_bExited(false)
#endif
   {
      op1 = &vmc->topData();
      op2 = op1 - 1;
   }

   /** Open an operator contet with a single operand.
    \param vmc The context of the virtual machine.
    \param op1 Where to store the first operand (top data).
    \param op2 Where to store the second operand (second-last).
    \param op3 Where to store the third operand (third-last).
    */
   inline OpToken( VMContext* vmc, Item*& op1, Item*& op2, Item*& op3 ):
      m_vmc( vmc ),
      m_ops( 2 )
#ifndef NDEBUG
      , m_bExited(false)
#endif
   {
      op1 = &vmc->topData();
      op2 = op1 - 1;
      op3 = op1 - 2;
   }


    /** Destructor.
    In debug, the destructor asserts if you forgot to call the exit()
    method.

    In release it's a no-op.
    */
   inline ~OpToken()
   {
      fassert2( m_bExited, "You forgot to exit from an open OpToken class");
   }

   /** Exits setting an operation value.
    \param value The value that the VM will see as the result of the operation.
    */
   inline void exit( const Item& value )
   {
#ifndef NDEBUG
      fassert2( ! m_bExited, "Already exited in OpToken class");
      m_bExited = true;
#endif
      m_vmc->popData(m_ops);
      fassert2( m_vmc->dataSize() >= 1, "Underflow in OpCount exit" );
      m_vmc->topData() = value;
   }

   /** Exits without setting an operation value.
    \note Use if you know what you're doing (TM), else use exit(const Item&)
    */
   inline void exit()
   {
#ifndef NDEBUG
      fassert2( ! m_bExited, "Already exited in OpToken class");
      m_bExited = true;
#endif
      m_vmc->popData(m_ops);
      fassert2( m_vmc->dataSize() >= 1, "Underflow in OpCount exit" );
   }

   /** Abandon the OpToken.
    This prevents the method to complain when leaving the function without having
    performed exit() (in debug mode -- in release this is ignored).

    This is usually done in multiple-step operations.
    Remember to give the VM its result via stepExit() or by correctly popping
    the items in the stack.
    */
   inline void abandon()
   {
#ifndef NDEBUG
      fassert2( ! m_bExited, "Already exited in OpToken class");
      m_bExited = true;
#endif      
   }

   /** Exits a deep step.
    This unrolls the current context stack and places the required result
    on top of it. It's to be considered a semantic closure of an OpToken being
    created at the begin of the first step.
    
    */
   inline static void stepExit( VMContext* vmc, int opCount, const Item& value )
   {
      vmc->popData(opCount-1);
      fassert2( vmc->dataSize() >= 1, "Underflow in OpCount exit" );
      vmc->topData() = value;
   }

   /** Exits a deep step.
    \note Use if you know what you're doing (TM).
    */
   inline static void stepExit( VMContext* vmc, int opCount )
   {
      vmc->popData(opCount-1);
      fassert2( vmc->dataSize() >= 1, "Underflow in OpCount exit" );
   }

};

}

#endif	/* _FALCON_OPTOKEN_H */

/* end of optoken.h */
