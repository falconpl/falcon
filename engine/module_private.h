/*
   FALCON - The Falcon Programming Language.
   FILE: module.cpp

   Falcon code unit
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Feb 2011 14:37:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_MODULE_PRIVATE_H
#define FALCON_MODULE_PRIVATE_H

#include <falcon/module.h>
#include <falcon/requirement.h>
#include <falcon/modrequest.h>

#include <deque>
#include <map>
#include <vector>
#include <set>

namespace Falcon {

class Module::Private
{
public:
   typedef std::deque<ModRequest*> ReqList;
   typedef std::set<Attribute*> AttribSet;
   
   /** List of requirements.
    
    This list is kept ordered as load order do matters when it comes to
    resolve import order and order of the execution of main code of modules.
    */
   ReqList m_mrlist;
   
   typedef std::map<String, ModRequest*> ReqMap;
   
   /** External modules static requirements, ordered by name.
    
    Name ordering is kept to simplify search for already define module 
    requirements.
    */
   ReqMap m_mrmap;
   
   /* Generic requirements, that is modules named under "import from Module".
    
    When an import request with a generic requirement is found, the ModRequest
    it bears is added here for simpler reference later on (so that it is not
    necessary to scan all the import requests to find them).
   */
   ReqList m_genericMods;
   
   typedef std::deque<ImportDef*> ImportDefList;
   
   /** List of import definitions.
    
    This is the list of "import" requests. It represents all the "import" directives,
    ordered by their appareance order in the module.
    
    In this list, both "direct" and source import requests are stored.
    
    Direct imports are imports directly generated from code and not from 
    a source script "import" or load directive.
    */
   ImportDefList m_importDefs; 
   
   //============================================
   // Requirements and dependencies
   //============================================
   
   /** Data representing a dependency from a foreign entity.
    
    Dependencies may be created through:
    - Explicit import requests (import directive).
    - Implicit import requests (undefined symbols).
    - Direct external symbol requirements.
    
    In the first two cases, an "extern" symbol in the host module gets bound to the
    dependency, and resolving the dependence means to update the symbol to 
    reference the external data.
    
    In case of direct dependency creation, the target symbol is just searched
    in the required module, when found, and the dependency calls the registered
    listeners (Requirement class instances).
    
    A dependency is generically bound to an import definition, unless it has
    been created by an implicit import. If bound to an import definition,
    it will be resolved in the scope of the module indicated by that, otherwise
    it will be resolved by searching the generically exported symbols.
    
    Dependencies created to import implicitly or explicitly symbols will be
    repeated in the m_depsBySymbol map. All the dependencies are stored in
    m_deps.
    */
   class Dependency
   {
   public:
      /** Local unresolved symbol.
       Could be zero in case of a dependency created by a direct request.
       */
      Variable* m_variable;
      
      /** The import definition related to this dependency.
       
       It will be nonzero if the symbol was created because of an explicit
       import or related to it.
       
       Could be zero if the dependency is created by an implicit impor request.
       */
      ImportDef* m_idef;
           
      /** The name of the symbol as it appears in the source module.
       It is calculated from the import definition, once applied namespaces
       and/or name aliasing defined there, or set to the same name
       of the implicitly imported symbol in case of implicit import.
       
       Dependencies created by direct imports requests will bear the name
       of the required symbol in the target module.
       */
      String m_sourceName;

      /** The name of the symbol as it is referenced in the target (local) module.
       */
      String m_targetName;

      /** We are responsible of the waiting list -- requirements are in the module. */
      typedef std::deque<Requirement*> WaitingList;
      
      /* Delayed requests waiting for this symbol to be resolved.
       
       This is a list of Requirement for things waiting for this dependency to be resolved.
       The requirements are generated 
       */      
      WaitingList m_waitings;
      
      /** Position of this entity in the module data.
       Used for simpler serialization, so that it is possible to reference this
       entity in the serialized file.
       */
      int m_id;

      int m_defLine;

      Dependency():
         m_variable( 0 ),
         m_idef( 0 ),
         m_defLine( 0 )
      {}
      
      Dependency( const String& name ):
         m_variable( 0 ),
         m_idef( 0 ),
         m_sourceName( name ),
         m_targetName( name ),
         m_defLine( 0 )
      {}
      
      Dependency( const String& srcName, Variable* sym, const String& tgName ):
         m_variable( sym ),
         m_idef( 0 ),
         m_sourceName( srcName ),
         m_targetName( tgName ),
         m_defLine( 0 )
      {}
      
      Dependency( const String& srcName, Variable* sym, const String& tgName, ImportDef* def ):
         m_variable( sym ),
         m_idef( def ),
         m_sourceName( srcName ),
         m_targetName( tgName ),
         m_defLine( 0 )
      {}

      ~Dependency() {}
      
      /** Called when the remote symbol is resolved. 
       \param sourceMod The module hosting this dependency
       \param hostMod The module where the symbol is imported, and this dependency is stored
       \param sourceVar The
       \return 0 if ok, a pointer to a composite error (error in a list)
       in case of errors.
       
       The symbol associated with this dependency, if any, is filled with
       the id and default value of the incoming symbol. Type is left untouched
       (and it should be left as "extern").
       
       This calls all the waiting functions, and fills the m_errors queue
       with errors returned by that functions, if any.
       */      
      Error* onResolved( Module* sourceMod, Module* hostMod, Item* source );
   };
   
   typedef std::deque<Dependency*> DepList;
   /** List of all the dependencies in the module.
    
    */
   DepList m_deplist;
   
   /** Map of dependencies by the explicit or implicit import symbol name. 
    
    Not all the dependencies are created because of an import. Some may be
    directly created by direct import requests.
    */   
   typedef std::map<String, Dependency*> DepMap;
   
   /* Map ext symbol name -> Dependency*.
    This map stores the dependecies known by this module. Each external symbol
    has a dependency which records:
    # Where it should be searched (an ImportDef referencing it).
    # What should be done when found.
    
    Both this informations can be null. If there isn't an ImportDef where it has
    to be searched, it is to be found in the global exports of the ModSpace
    where this module resides.
    */
   DepMap m_depsByName;
   
   /** Class used to keep track of requests to import a whole namespace.
    Namespace imports are request to import multiple symbols in the owner module.
    
    They are created throught the "import/ * /from" directive, where * indicates
    the whole of a namespace. 
    
    Namespace imports are always bound to an import definition that
    asks for multiple symbols to be imported.
    
    This class is used to keep track of where this kind of import directive
    is meant to perform the transfer.
    */
   class NSImport {
   public:
      ImportDef* m_def;
      String m_from;
      String m_to;
      bool m_bPerformed;
      
      NSImport( ImportDef* def, const String& from, const String& to ):
         m_def( def ),
         m_from( from ),
         m_to( to ),
         m_bPerformed( false )
      {}
   };
   
   typedef std::deque<NSImport*> NSImportList;
   NSImportList m_nsimports;
   
      
   //============================================================
   // Visible entities
   //

   typedef std::map<String, Mantra*> MantraMap;
   MantraMap m_mantras;

   typedef std::deque<Requirement*> RequirementList;
   RequirementList m_reqslist;
   
   // used during deserialization
   uint32 m_symCount;
   std::vector<int32> m_tempExport;
   
   typedef std::deque<Class*> InitList;
   InitList m_initList;

   Private()
   {}

   ~Private();
};

}

#endif	/* MODULE_PRIVATE_H */

/* end of module_priavate.h */
