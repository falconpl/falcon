/*
   FALCON - The Falcon Programming Language.
   FILE: requiredclass.h

   Structure holding information about classes needed elsewhere.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_REQUIREMENT_H_
#define _FALCON_REQUIREMENT_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

class Error;
class Module;
class Symbol;


/** Functionoid for delayed resolution of symbols.
 */
class FALCON_DYN_CLASS Requirement
{
public:
   Requirement( const String& name ) :
      m_name( name )
   {}
      
   virtual ~Requirement() {}
   
   /** Called back when the requirement is resolved.
    \param source The module where the requirement is found -- might be 0 if 
           generated internally.
    \param srcSym The symbol answering the requirement.
    \param tgt The module where the symbol was searched. Might be zero if not
            invoked within a module.
    \param extSym The symbol representing the external reference in the target
            module.
    \throw A CodeError may be thrown by the implementation to indicate some kind of
          rule being broken (e.g. the requirement for inheritances must be classes).

    
    The source module can be 0 if the symbol is internally generated.
    
    The target module and extSym can be 0 if the system resolves the dependency on its own.
    
    The source and target modules can be the same module if a requirement is resolved by
    a later definition. In this case, extSym and srcSym will also point to the same symbol.
    */
   virtual void onResolved( const Module* source, const Symbol* srcSym, Module* tgt, Symbol* extSym ) = 0;

   const String& name() const { return m_name; }
   
private:
   String m_name;
};

}

#endif /* _FALCON_REQUIREMENT_H_ */

/* end of requiredclass.h */
