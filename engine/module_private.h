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

#include <deque>
#include <list>
#include <map>

namespace Falcon {

class Module::Private
{
public:
   typedef std::deque<ImportDef*> ImportDefList;
   
   //============================================
   // Requirements and dependencies
   //============================================
   
   /** Class keeping records of modules requested (referenced) by this module. */
   class ModRequest
   {
   public:
      String m_name;
      bool m_isLoad;
      bool m_bIsURI;
      Module* m_module;
      
      ImportDefList m_defs;
      
      ModRequest();
      ModRequest( const String& name, bool isUri = false, bool isLoad = false, Module* mod = 0);
      
      ~ModRequest();
   };
   
   /** Record keeping track of needed modules (eventually already defined). */
   class Dependency
   {
   public:
      /** Local unresolved symbol. */
      Symbol* m_symbol;
      
      /** The import definition related to this dependency. 
       It will be nonzero if the symbol was created because of an explicit
       import or related to it.
       */
      ImportDef* m_idef;
      
      /** The symbol designated by this dependency once resolved. */
      Symbol* m_resSymbol;
      
      /** The module request where the source module for this symbol is to be resolved (if direct) */
      ModRequest* m_sourceReq; 
      
      /** The name of the symbol as it appares in the source module. */
      String m_sourceName;

      typedef std::list<Requirement*> WaitingList;
      
      /* Delayed requests waiting for this synmbol to be resolved. 
       
       This is a list of Requirement for things waiting for this dependency to be resolved.
       The requirements are generated 
       */      
      WaitingList m_waitings;

      Dependency():
         m_symbol( 0 ),
         m_idef( 0 ),
         m_resSymbol( 0 ),
         m_sourceReq(0)
      {}
      
      Dependency( Symbol* sym ):
         m_symbol( sym ),
         m_idef( 0 ),
         m_resSymbol( 0 ),
         m_sourceReq(0)
      {}
      
      Dependency( Symbol* sym, ImportDef* def ):
         m_symbol( sym ),
         m_idef( def ),
         m_resSymbol( 0 ),
         m_sourceReq(0)
      {}

      ~Dependency();
      
      /** Called when the remote symbol is resolved. 
       \param parentMod The module hosting this dependency
       \param mod The module where the symbol is imported (not exported!)
       \param sym the defined symbol (coming from the exporter module).
       \return 0 if ok, a pointer to a composite error (error in a list)
       in case of errors.
       
       The symbol associated with this dependency, if any, is filled with
       the id and default value of the incoming symbol. Type is left untouched
       (and it should be left as "extern").
       
       This calls all the waiting functions, and fills the m_errors queue
       with errors returned by that functions, if any.
       */      
      Error* onResolved( Module* parentMod, Module* mod, Symbol* sym );
   };
   
   
   //============================================
   // Main data
   //============================================

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
   DepMap m_deps;
   
   class DirectRequest 
   {
   public:
      /** The import definition related to this dependency. 
       */
      ImportDef* m_idef;      
      t_func_import_req m_cb;
      
      DirectRequest() {}
      
      DirectRequest( ImportDef* idef, t_func_import_req cb ):
         m_idef( idef ),
         m_cb(cb)
      {}
      
      ~DirectRequest();
   };
   
   typedef std::deque<DirectRequest *> DirectReqList;
   DirectReqList m_directReqs;
   
   
   ImportDefList m_importDefs;   
   
   typedef std::map<String, ModRequest*> ReqMap;
   
   /** External modules static requirements.         
    */
   ReqMap m_mrmap;
   
   typedef std::deque<ModRequest*> ReqList;
   /** List of requirements.
    This list is kept ordered as load order do matters when it comes to
    resolve import order and order of the execution of main code of modules.
    */
   ReqList m_mrlist;
   
   // Generic requirements, that is, import from Module 
   // (modules that might provide any undefined symbol).
   ReqList m_genericMods;

   typedef std::map<String, Symbol*> GlobalsMap;
   GlobalsMap m_gSyms;
   GlobalsMap m_gExports;

   typedef std::map<String, Function*> FunctionMap;
   FunctionMap m_functions;

   typedef std::map<String, Class*> ClassMap;
   ClassMap m_classes;

   typedef std::list<Item> StaticDataList; 
   StaticDataList m_staticData;
   
   class NSImport {
   public:
      ImportDef* m_def;
      ModRequest* m_req;
      String m_from;
      String m_to;
      bool m_bPerformed;
      
      NSImport( ImportDef* def, ModRequest* mr, const String& from, const String& to ):
         m_def( def ),
         m_req( mr ),
         m_from( from ),
         m_to( to ),
         m_bPerformed( false )
      {}
   };
   
   typedef std::deque<NSImport*> NSImportList;
   NSImportList m_nsimports;
   
   Private()
   {}

   ~Private();
};

}

#endif	/* MODULE_PRIVATE_H */

/* end of module_priavate.h */
