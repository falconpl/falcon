/*
   FALCON - The Falcon Programming Language.
   FILE: synfunc.h

   Function objects -- expanding to new syntactic trees.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_SYNFUNCTION_H_
#define FALCON_SYNFUNCTION_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/function.h>
#include <falcon/syntree.h>

namespace Falcon
{

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

class FALCON_DYN_CLASS SynFunc: public Function
{
public:
   SynFunc( const String& name, Module* owner = 0, int32 line = 0 );
   virtual ~SynFunc();    
   
   /** Returns the statements of this function */
   const SynTree& syntree() const { return m_syntree; }
   SynTree& syntree() { return m_syntree; }

   virtual void apply( VMachine* vm, int32 pCount = 0 );
   
protected:
   SynTree m_syntree;
};

}

#endif /* SYNFUNCTION_H_ */

/* end of synfunc.h */
