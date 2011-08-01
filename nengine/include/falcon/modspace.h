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

namespace Falcon {

class VMachine;
class Module;
class Error;
class Symbol;

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
   ModSpace();
   virtual ~ModSpace();
   
   VMachine* vm() const { return m_vm;  }

   /** Adds a link error.
    \param err_id Id of the error.
    \param mod The module where the error was found -- can be 0.
    \param sym The symbol that caused the error.
    \param extra Extra description.

    During the link process, multiple errors could be found.
    When the link process is complete, the Virtual Machine owner will
    call checkRun() that will throw an error.    
    */
   void addLinkError( int err_id, const String& modname, const Symbol* sym, const String& extra="" );

   /** Adds a link error.
    \param e The error to be added.
    */
   void addLinkError( Error* e );
   
   /** Returns all the generated error as a single composed error. 
    \return a LinkError of type e_link_error containing all the suberrors, 
    or 0 if there was no error to be returned.
    
    Once called, the previusly recorded errors are cleared.
    */
   Error* makeError() const;
   
   /** Finds a module in the space structure.
    \param local_name The name of the module to be found.
    \param isLoad Will be set to true if the module is required for load.
    \return A valid module or 0 if not found.    
    
    About the \b load paramter, see promoteLoad() method.
    */
   Module* findModule( const String& local_name, bool &isLoad ) const;
   
   /** Finds a module in the space structure.
    \param local_name The name of the module to be found.
    \return A valid module or 0 if not found.    
    
   
    */
   Module* findModule( const String& local_name ) const
   {
      bool dummy;
      return findModule( local_name, dummy );
   }
   
   /** Promotes a previously existing module to load.
    \param local_name The name of the module to be promoted.
    \return True if the module was promoted, false if it doesn't exist or already
            stored for load.
    
    Loaded modules differ from imported modules as imported ones are explicitly
    queried for symbols by importers. So, if a module depends on an imported one,
    that module is added to this module space for import. 
    
    If another module which is then loaded or imported afterwards requires to
    load the same module that was previusly required for import, the subject
    module is "promoted" to load.
    
    It is safe to apply the promoteLoad() on all the already existing modules
    that are declared for load.
    */
   bool promoteLoad( const String& local_name );
   
   /** Adds a module to the module space.
    \param mod The module to be linked.
    \param isLoad If true, the module is required for load.
    \return true on success, false if there is already a module with the same name.
    
    This adds the module in this ModSpace. The added module is not immediately
    linked, nor initialized. Those operations are perfomed after the link() and
    initialize() method calls.
    
    If a module is required for load, then its export directives are honoured
    and the symbol declared as exportable are actually exported. Otherwise,
    the module is considered as added for "import/from", and any symbol
    eventually exported is ignored.
    
    A module previously added for import can be promoted to added for load
    later on, through the promoteLoad() method.
    
    \note The caller must have already provided the module with a name
    consistent with the internal naming scheme.
    */
   bool addModule( Module* mod, bool isLoad );
   
   /** Links the previously added modules. 
    \return true on success, false in case of errors during the link phase.
    
    The link process happens in two steps: first, all the modules added as
    "load" subjects are queried for exported symbols last to first. If a 
    module that was added \b before another one exports a symbol that was 
    already exported by another exporter, an error is added.
    
    After all the exports are honoured, the modules are queried for external
    symbols which are then resolved. If some required symbols are not found,
    an error is added.
    
    The method returns true if the link process could complete without any error,
    and false if there was some error during the link process.
    */
   bool link();
   
   /** Prepares the invocation of initialization methods.
    This method is to be called after link() and before the virtual machine
    is finally launched for run.
    */
   void readyVM( VMachine* vm );
   
   
   /** Finds a symbol that is globally exported or globally defined.
    \param name The name of the symbol that is exported.
    \param declarer A pointer that will be set with the module where the symbol was declared.
    \return The symbol, if defined or 0 if the name cannot be found.
    
    \note The caller must be prepared to the event that a symbol is found, but
    the \b declarer parameter is set to zero. In fact, it is possible for embedding
    applications to create module-less symbols.
    */
   const Symbol* findExportedSymbol( const String& name, Module*& declarer ) const;
   
   /** Finds a symbol that is globally exported or globally defined.
    \param name The name of the symbol that is exported.
    \return The symbol, if defined or 0 if the name cannot be found.
    This version doesn't return the module where the symbol was declared.
    */
   
   const Symbol* findExportedSymbol( const String& name ) const
   {
      Module* dummy;
      return findExportedSymbol( name, dummy );
   }

   /** Adds a symbol to the exported map. 
    \param mod The module exporting the symbol.
    \param sym The symbol being exported.
    \param bAddError if true, add an alreadydef error in case of symbol already defined.
    \return true on success, false if the symbol was already exported.
    
    This method is mainly meant to be used during the link() process. However,
    it's legal to pre-export symbols, and eventually associate a null module
    to symbols created by the application.
    */
   bool addExportedSymbol( Module* mod, const Symbol* sym, bool bAddError = false );
    
private:
   VMachine* m_vm;
    
   class Private;
   Private* _p;
   
   void link_exports();
   void link_imports();
};

}

#endif

/* end of modspace.h */
