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

#define FALCON_VM_DFAULT_CHECK_LOOPS 5000

namespace Falcon {

class VMachine;
class Module;

/** Collection of (static) modules active in a virtual machine.
 
 The module place is a place where modules reside and possibly share their
 data through the standard import-export directives.
 
 A module can be statically or dynamically linked against a module space.
 Static modules are added to the space and can be queried for data later on.
 Dynamic modules just read data from the space without sending anything to it,
 and publish their contents directly to the virtual machine (and to the scripts)
 performing explicit requests.

*/
class FALCON_DYN_CLASS ModSpace
{

public:
   /** Creates the module space on the given virtual machine.*/
   ModSpace( VMachine* owner );
   virtual ~ModSpace();
   
   VMachine* vm() const { return m_vm;  }

   /** Finds a symbol that is globally exported or globally defined.
    \return The symbol, if defined or 0 if the name cannot be found.
    \param name The name of the symbol that is exported.
    
    */
   const Symbol* findExportedSymbol( const String& name ) const;

   /** Adds a symbol to the exported map. 
    \param mod The module exporting the symbol.
    \param sym The symbol being exported.
    */
   bool addExportedSymbol( Module* mod, const Symbol* sym );

   /** Adds a link error.
    \param err_id Id of the error.
    \param mod The module where the error was found.
    \param sym The symbol that caused the error.
    \param extra Extra description.

    During the link process, multiple errors could be found.
    When the link process is complete, the Virtual Machine owner will
    call checkRun() that will throw an error.    
    */
   void addLinkError( int err_id, Module* mod, const Symbol* sym, const String& extra="" );

   /** Adds a module to the module space.
    \parm local_name The local name under which the module is known.
    \param mod The module to be linked.
    \return true on success, false on error.
    
    This links the module in this ModSpace. In case of errors, the enumerateErrors
    can be invoked to 
    */
   bool link( const String& local_name, Module* mod );
   
private:
   VMachine* m_vm;
   
   class Private;
   Private* _p;
};

}

#endif

/* end of modspace.h */
