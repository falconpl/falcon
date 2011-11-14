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

#include <deque>
#include <list>
#include <map>

namespace Falcon {

class Module::Private
{
public:
   //============================================
   // Requirements and dependencies
   //============================================

   class WaitingDep
   {
   public:
      virtual Error* onSymbolLoaded( Module* mod, Symbol* sym ) = 0;
   };
   
   class WaitingFunc: public WaitingDep
   {
   public:
      Module* m_requester;
      t_func_import_req m_func;

      WaitingFunc( Module* req, t_func_import_req func ):
         m_requester( req ),
         m_func( func )
      {}
      
      virtual Error* onSymbolLoaded( Module* mod, Symbol* sym );
   };

   class WaitingInherit: public WaitingDep
   {
   public:
      Inheritance* m_inh;

      WaitingInherit( Inheritance* inh ):
         m_inh( inh )
      {}

      virtual Error* onSymbolLoaded( Module* mod, Symbol* sym );
   };
   
   class WaitingRequirement: public WaitingDep
   {
   public:
      Requirement* m_cr;

      WaitingRequirement( Requirement* inh ):
         m_cr( inh )
      {}

      virtual Error* onSymbolLoaded( Module* mod, Symbol* sym );
   };

   /** Records a single remote name with multiple items waiting for that name to be resolved.*/
   class Dependency
   {
   public:
      String m_remoteName;
      /** Local symbol representing the unkonwn remote symbol.
       This is just a shortcut storage for the symbol, which is actually
       stored in the globals of this module, under the localName of this
       symbol (which is in the DepMap to which this dependency belongs).
       
       It might be if there is an explicit import request from an extension
       module.
       */
      Symbol* m_symbol;
      
      Symbol* m_resolvedSymbol;
      Module* m_resolvedModule;
      
      typedef std::deque<WaitingDep*> WaitList;
      WaitList m_waiting;
      
      typedef std::deque<Error*> ErrorList;
      ErrorList m_errors;

      Dependency( const String& rname ):
         m_remoteName( rname ),
         m_symbol(0),
         m_resolvedSymbol(0)
      {}

      ~Dependency();
      
      void clearErrors();
      
      /** Called when the remote name is resolved. 
       \param Module The module where the symbol is imported (not exported!)
       \parma sym the defined symbol (coming from the exporter module).
       
       The symbol associated with this dependency, if any, is filled with
       the id and default value of the incoming symbol. Type is left untouched
       (and it should be left as "extern").
       
       This calls all the waiting functions, and fills the m_errors queue
       with errors returned by that functions, if any.
       */
      void resolved( Module* mod, Symbol* sym );
      
      Error* resolveOnModSpace( ModSpace* ms, const String& uri, int line );
   };

   /** Record keeping track of needed modules (eventually already defined). */
   class Request
   {
   public:
      String m_uri;
      bool m_bIsUri;
      t_loadMode m_loadMode;
      /** This pointer stays 0 until resolved via resolve[*]Reqs */
      Module* m_module;

      // Local name -> remote dependency
      typedef std::map<String, Dependency*> DepMap;
      DepMap m_deps;
      
      bool m_bIsGenericProvider;
      
      // Remote NS (or "") -> local NS (or "")
      typedef std::map<String, String> NsImportMap;
      NsImportMap m_fullImport;

      Request( const String& name, t_loadMode imode=e_lm_import_public, bool bIsUri=false ):
         m_uri( name ),
         m_bIsUri( bIsUri ),
         m_loadMode( imode ),
         m_module(0),
         m_bIsGenericProvider( false )
      {}

      ~Request();      
   };
   
   /** Class used to keep track of full remote namespace imports.
    */
   class NSImport {
   public:
      String m_remNS;
      Request* m_req;
      
      NSImport( const String& remNS, Request* req ):
      m_remNS( remNS ), m_req(req)
      {}
         
   };
  
   //============================================
   // Main data
   //============================================

   // Map module-name/path -> requirent*
   typedef std::map<String, Request*> ReqMap;
   ReqMap m_reqs;
   
   typedef std::map<String, NSImport*> NSImportMap;
   NSImportMap m_nsImports;
   
   typedef std::deque<Request*> ReqList;
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
   
   typedef std::deque<Inheritance*> InheritanceList;
   InheritanceList m_pendingInh;
   
   Private()
   {}

   ~Private();

   Dependency* getDep( const String& sourcemod, bool bIsUri, const String& symname, bool bSearchNS = false );   
   /** Finds an existing dependency in an existing module.
    */
   Dependency* findDep( const String& sourcemod, const String& symname ) const;   
   /** Erases a dependency from a source module request. 
    \param sourcemod The module originally requested for this symbol name.
    \param symname The symbol name (eventually complete with the local namespace.
    \param clearReq If true and this was the last dependency in the request, removes the reqest as well.
    */
   void removeDep( const String& sourcemod, const String& symname, bool clearReq=false );   
   Request* getReq( const String& sourcemod, bool bIsUri );
      
   /** Prepares the whole namespace import
    */
   bool addNSImport( const String& localNS, const String& remoteNS, Request* req );
   
   /** Adds a requsirement to the dependency list. */
   Dependency* addRequirement( Requirement * req );
};

}

#endif	/* MODULE_PRIVATE_H */

/* end of module_priavate.h */
