/*
   FALCON - The Falcon Programming Language.
   FILE: modgroup.h

   Group of modules involved in a single link operation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MODGROUP_H_
#define _FALCON_MODGROUP_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/modgroupbase.h>
#include <falcon/loadmode.h>

namespace Falcon {

class VMContext;
class Module;
class Error;
class Symbol;
class ModLoader;
class ModSpace;

/** Group of modules involved in a single link operation.
 
 The request to launch a foreign module can evolve into more reuquests
 for more modules. Requests to export symbols or publish new
 modules to the virtual machine are pre-processed during the link
 stage through this class.
 
 See ModSpace for more information about module loading and linking.

*/
class FALCON_DYN_CLASS ModGroup: public ModGroupBase
{

public:
   
   /** Creates the module space on the given virtual machine.*/
   ModGroup( ModSpace* owner );
   virtual ~ModGroup();

   /** Adds a module to this module group, resolving its dependencies.
    \param m The module to be added.
    */
   bool add( Module* mod, t_loadMode mode );
   
   void gcMark( uint32 mark ) { m_lastGCMark = mark; }
   
   uint32 lastGCMark() const { return m_lastGCMark; }
   
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
    
   
   /** Links the previously added modules onto a final module space. 
    \return true on success, false in case of errors during the link phase.
    
    Modules previously resolved through add() get linked (their foreign symbols
    get resolved) during this step.
    */
   bool link();
   
   /** Prepares the invocation of initialization methods.
    \param ctx The context where to prepare the initialization of the group.
    
    This method is to be called after link() and before the virtual machine
    is finally launched for run.
    */
   void readyVM( VMContext* ctx );   
   
   /** Get the space in which this group resides. */
   ModSpace* space() const { return m_owner; }
   
private:
   class Private;
   Private* _p;
   
   ModSpace* m_owner;
   uint32 m_lastGCMark;
   
   bool linkDirectImports();
   bool linkExports( bool bForReal );
   bool linkGenericImports( bool bForReal );
   void commitModules( bool bForReal );   

};

}

#endif

/* end of modgroup.h */
