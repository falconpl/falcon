/*
   FALCON - The Falcon Programming Language.
   FILE: modspace.h

   Module space for the Falcon virtual machine
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_MODSPACE_H
#define FALCON_MODSPACE_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/modgroupbase.h>

namespace Falcon {

class VMachine;
class VMContext;
class Module;
class Error;
class Symbol;
class ModLoader;

/** Collection of (static) modules active in a virtual machine.
 
 The module place is a place where modules reside and possibly share their
 data through the standard import-export directives.
 
 A module can be statically or dynamically linked against a module space.
 Static modules are added to the space and can be queried for data later on.
 Dynamic modules just read data from the space without sending anything to it,
 and publish their contents directly to the virtual machine (and to the scripts)
 performing explicit requests.

*/
class FALCON_DYN_CLASS ModSpace: public ModGroupBase
{

public:
   
   /** Creates the module space on the given virtual machine.*/
   ModSpace( VMachine* vm );
   virtual ~ModSpace();
   
   VMachine* vm() const { return m_vm; }
    
   /** Gets the module loader that's associated with this module space.
    \return 
    
    The module space uses a modLoader to resolve automatically the dependencies
    of added modules.
    */
   ModLoader* modLoader() const { return m_loader; }

   /** Adds a token declaringa future symbol.
    \param name The name of the publically exported symbol that will be added afterwards.
    \param modName Symbolic name or URI of the module declaring.
    \param declarer Filled with the previous declarer if a token was already set.
    \return true if the symbol can be reserved, false if the symbol was already
    reserved by some other module.
    */
   bool addSymbolToken( const String& name, const String& modName, String& declarer );
   
   /** Retire a symbol token.
    \param name The name of the publically exported symbol that will be added afterwards.
    \param modName Symbolic name or URI of the module declaring.
    */
   void removeSymbolToken( const String& name, const String& modName );
   
   Symbol* findExportedSymbol( const String& name ) const;
   
   bool addSymbol( Symbol* sym, Module* mod = 0 );
   
   Error* add( Module* mod, t_loadMode lm, VMContext* ctx );

private:
   VMachine* m_vm;

   class Private;
   Private* _p;
   
   ModLoader* m_loader;
   
   void exportSymbols( Module* mod );
};

}

#endif

/* end of modspace.h */
