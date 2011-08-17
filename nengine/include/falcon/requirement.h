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
#include <falcon/sourceref.h>

namespace Falcon
{

class Requirement;
class Error;
class Symbol;

/** Class used to callback a symbol requirer.
 
 \see Requirement.
 */
class FALCON_DYN_CLASS Requirer
{
public:
   virtual ~Requirer() {}
   /** Callback issued when a requirement gets resolved.
    \param source The module where the symbol was declared -- might be 0 if generated
      internally!
    \param sym The symbol resolving the requirement.
    \param sender The requirement that got resolved by this symbol.
    \return 0 if ok, an Error* instance on problem.
    
    If the requirer is not satisfied by this symbol i.e. because of an unexpected
    type, it can have the linker to add an error by returning a new error instance.
    If the requirement was correctly completed, just return 0.
    */
   virtual Error* resolved( Module* source, const Symbol* sym, Requirement* sender ) = 0;
};


/** Structure holding a reference to an item that should be filled during link.
 
 Similar to Inheritance, this class is used by statements requiring items
 that may come from outside a module, but that are not class declaration 
 (catch and select are two examples).
 
 Once resolved, the required is called back and can perform "link completion".
 
 The requirer will get a notification through the Requirer::resolved functionoid.
 
 Subclasses may be created to allow storing other data that is needed by the
 requirer at resolution.
 
 \note The requirement class ownership stays on the requirer. They are never
 destroyed by the linker or by other engine elements, so they can be held
 by the requirer and be used after they get resolved.
 */
class FALCON_DYN_CLASS Requirement
{
public:
   Requirement( const String& name, Requirer* req );
   ~Requirement();

   /** The name of the symbol that we're searching.
    This includes full path/namespace.

    The required name is the name of the symbol as it's written
    in the statement requiring it.
    */
   const String& name() const { return m_name; }

   /** Gets the requirer associated with this requirement.
    \return The associated requirer.
    */
   Requirer* requirer() const { return m_req; }
   
   /** Resolves the requirement.
    \param mod The module where the requirement is found -- might be 0 if generated internally.
    \param sym The symbol answering the requirement.
    */
   Error* resolve( Module* mod, const Symbol* sym );
   
   SourceRef& sourceRef() { return m_sr; }
   const SourceRef& sourceRef() const { return m_sr; }
   
private:
   String m_name;
   Requirer* m_req;
   SourceRef m_sr;
};

}

#endif /* _FALCON_REQUIREMENT_H_ */

/* end of requiredclass.h */
