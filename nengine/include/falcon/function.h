/*
   FALCON - The Falcon Programming Language.
   FILE: function.h

   Function objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FUNCTION_H_
#define FALCON_FUNCTION_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

class Collector;
class VMachine;

/**
 Falcon function.

 This class represents the minimal execution unit in Falcon. It's a set of
 code (to be excuted), symbols (parameters, local variables and reference to
 global variables in the module) and possibly closed values.

 Functions can be directly executed by the virtual machine.

 They usually reside in a module, of which they are able to access the global
 variable vector (and of which they keep a reference).

 To achieve higher performance, functions are not treated as
 normal garbageable items (the vast majority of them is never really
 destroyed). They become garbageable when their module is explicitly
 unloaded while linked, or when they are created dynamically as closures,
 or when constructed directly by the code.

 Functions can be created by modules or directly from the code. In this case,
 they aren't owned by any module and are immediately stored for garbage collection.
*/

class FALCON_DYN_CLASS Function
{
public:
   Function( const String& name, Module* owner = 0, int32 line = 0 );
   virtual ~Function();
   
   /** Sets the module of this function.
    Mainly, this information is used for debugging (i.e. to know where a function
    is declared).
    */
   void module( Module* owner );

   /** Return the module where this function is allocated.
   */
   Module* module() const { return m_module; }

   /** Returns the name of this function. */
   const String& name() const { return m_name; }

   /** Returns the source line where this function was declared.
    To be used in conjunction with module() to pinpoint the location of a function.
    */
   int32 declaredAt() const { return m_line; }

   /** Returns the complete identifier for this function.

    The format is:
         name(line) source/path.fal

    Where a source path is not available, the module full qualified name is used.
   
   */
   String locate() const;   

   int32 paramCount() const { return m_paramCount; }

   /** Sets the variable count.
    * @param pc The number of parameters in this functions.
    *
    * the parameter count should be a number in 0..varCount().
    */
   void paramCount( int32 pc ) { m_paramCount = pc; }

   /** Mark this function for garbage collecting. */
   void gcMark( int32 mark );

   /** Store in a garbage collector. 
    
    When this method is called, the function become subject to garbage
    collection.
   
    */
   GCToken* garbage( Collector* c );

   /** Garbage this function on the standard collector. */
   GCToken* garbage();

   /** Executes the call.

    The call execution may be either immediate or deferred; for example,
    the call may just leaves PSteps to be executed by the virtual machine.

    In case of deferred calls, apply must also push proper return PStep codes.
    In case of immediate calls, apply() must also perform the return frame
    code in the virtual machine by calling VMachine::returnFrame().
    */
   virtual void apply( VMachine* vm, int32 pCount = 0 ) = 0;

   /** Just candy grammar for this->apply(vm); */
   void operator()( VMachine* vm ) { apply(vm); }

protected:
   String m_name;
   int32 m_paramCount;

   GCToken* m_gcToken;   
   Module* m_module;

   int32 m_line;
};

}

#endif /* FUNCTION_H_ */

/* end of function.h */
