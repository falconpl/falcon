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

#undef SRC
#define SRC "engine/module.cpp"

#include <falcon/trace.h>
#include <falcon/module.h>
#include <falcon/itemarray.h>
#include <falcon/globalsymbol.h>
#include <falcon/unknownsymbol.h>
#include <falcon/extfunc.h>
#include <falcon/item.h>
#include <falcon/modspace.h>
#include <falcon/modloader.h>
#include <falcon/inheritance.h>
#include <falcon/error.h>
#include <falcon/linkerror.h>
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/dynunloader.h>

#include <stdexcept>
#include <map>
#include <deque>

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

   class WaitingSym: public WaitingDep
   {
   public:
      UnknownSymbol* m_symbol;

      WaitingSym( UnknownSymbol* sym ):
         m_symbol( sym )
      {}

      virtual Error* onSymbolLoaded( Module*, Symbol* sym )
      {
         m_symbol->define( sym );
         return 0;
      }
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
      
      virtual Error* onSymbolLoaded( Module* mod, Symbol* sym )
      {
         return m_func( m_requester, mod, sym );
      }
   };

   class WaitingInherit: public WaitingDep
   {
   public:
      Inheritance* m_inh;

      WaitingInherit( Inheritance* inh ):
         m_inh( inh )
      {}

      virtual Error* onSymbolLoaded( Module* mod, Symbol* sym )
      {
         Item* value;
         if( (value = sym->value( 0 )) == 0 || ! value->isClass() )
         {
            // the symbol is not global?            
            return new LinkError( ErrorParam( e_inv_inherit ) 
               .module(mod->name())
               .symbol( m_inh->owner()->name() )
               .line(m_inh->sourceRef().line())
               .chr(m_inh->sourceRef().chr())
               .extra( sym->name() )
               .origin(ErrorParam::e_orig_linker));
         }
         
         // Ok, we have a valid class.
         Class* parent = static_cast<Class*>(value->asInst());
         m_inh->parent( parent );
         // is the class a Falcon class?
         if( parent->isFalconClass() )
         {
            // then, see if we can link it.
            FalconClass* falcls = static_cast<FalconClass*>(parent);
            if( falcls->missingParents() == 0 )
            {
               mod->completeClass( falcls );
            }
         }
         return 0;
      }
   };

   /** Records a single remote name with multiple items waiting for that name to be resolved.*/
   class Dependency
   {
   public:
      String m_remoteName;
      
      typedef std::deque<WaitingDep*> WaitList;
      WaitList m_waiting;
      
      typedef std::deque<Error*> ErrorList;
      ErrorList m_errors;

      Dependency( const String& rname ):
         m_remoteName( rname )
      {}

      ~Dependency()
      {
         // the symbol is owned by the global map.
         WaitList::iterator wli = m_waiting.begin();
         while( wli != m_waiting.end() )
         {
            delete *wli;
            ++wli;
         }
         
         clearErrors();
      }

      
      void clearErrors() {
         ErrorList::iterator eli = m_errors.begin();
         while( eli != m_errors.end() )
         {
            (*eli)->decref();
            ++eli;
         }
         m_errors.clear();
      }
      
      /** Called when the remote name is resolved. 
       \param Module The module where the symbol is imported (not exported!)
       \parma sym the defined symbol (coming from the exporter module).
       */
      void resolved( Module* mod, Symbol* sym )
      {
         WaitList::iterator iter = m_waiting.begin();
         while( iter != m_waiting.end() )
         {
            WaitingDep* dep = *iter;
            Error* err = dep->onSymbolLoaded( mod, sym );
            if( err != 0 )
            {
               m_errors.push_back( err );
            }
            ++iter;
         }
      }
   };

   /** Record keeping track of needed modules (eventually already defined). */
   class Requirement
   {
   public:
      String m_uri;
      bool m_bIsUri;
      bool m_bIsLoad;
      /** This pointer stays 0 until resolved via resolve[*]Reqs */
      Module* m_module;
      /** Dinamicity of modules determine if they are to be deleted.*/
      bool m_bIsDynamic;

      // Local name -> remote dependency
      typedef std::map<String, Dependency*> DepMap;
      DepMap m_deps;
      
      bool m_bIsGenericProvider;
      
      // Remote NS (or "") -> local NS (or "")
      typedef std::map<String, String> NsImportMap;
      NsImportMap m_fullImport;

      Requirement( const String& name, bool bIsLoad, bool bIsUri=false ):
         m_uri( name ),
         m_bIsUri( bIsUri ),
         m_bIsLoad( bIsLoad ),
         m_module(0),
         m_bIsDynamic( false ),
         m_bIsGenericProvider( false )
      {}

      ~Requirement()
      {
         DepMap::iterator dep_i = m_deps.begin();
         while( dep_i != m_deps.end() )
         {
            delete dep_i->second;
            ++dep_i;
         }
         
         if( m_bIsDynamic )
         {
            delete m_module;
         }
      }
   };
   
   /** Class used to keep track of full remote namespace imports.
    */
   class NSImport {
   public:
      String m_remNS;
      Requirement* m_req;
      
      NSImport( const String& remNS, Requirement* req ):
      m_remNS( remNS ), m_req(req)
      {}
         
   };
   
   class SymbolGuard {
   public:
      Symbol* m_sym;
      bool m_bOwn;
      
      SymbolGuard( Symbol* sym=0, bool bOwn = true ):
         m_sym( sym ),
         m_bOwn( bOwn )
      {}
   };
   //============================================
   // Main data
   //============================================

   // Map module-name/path -> requirent*
   typedef std::map<String, Requirement*> ReqMap;
   ReqMap m_reqs;
   
   typedef std::map<String, NSImport*> NSImportMap;
   NSImportMap m_nsImports;
   
   typedef std::deque<Requirement*> ReqList;
   // Generic requirements, that is, import from Module 
   // (modules that might provide any undefined symbol).
   ReqList m_genericMods;

   typedef std::map<String, Symbol*> GlobalsMap;
   typedef std::map<String, SymbolGuard> SymbolGuardMap;
   SymbolGuardMap m_gSyms;
   GlobalsMap m_gExports;

   typedef std::map<String, Function*> FunctionMap;
   FunctionMap m_functions;

   typedef std::map<String, Class*> ClassMap;
   ClassMap m_classes;

   ItemArray m_staticdData;


   Private()
   {}

   ~Private()
   {
      // We can destroy the globals, as we're always responsible for that...
      SymbolGuardMap::iterator iter = m_gSyms.begin();
      while( iter != m_gSyms.end() )
      {
         if( iter->second.m_bOwn )
         {
            Symbol* sym = iter->second.m_sym;
            delete sym;
         }
         ++iter;
      }

      // and get rid of the static data, if we have.
      for ( length_t is = 0; is < m_staticdData.length(); ++ is )
      {
         Item& itm = m_staticdData[is];
         itm.asClass()->dispose( itm.asInst() );
      }

      // destroy reqs and deps
      ReqMap::iterator req_i = m_reqs.begin();
      while( req_i != m_reqs.end() )
      {
         delete req_i->second;
         ++req_i;
      }

      // we can always delete the classes that have been assigned to us.
      ClassMap::iterator cli = m_classes.begin();
      while( cli != m_classes.end() )
      {
         // set the module to 0, so that we're not dec-reffed.
         cli->second->detachModule();
         delete cli->second;
         ++cli;
      }

      FunctionMap::iterator vi = m_functions.begin();
      while( vi != m_functions.end() )
      {
         // set the module to 0, so that we're not dec-reffed.
         vi->second->detachModule();
         // then delete the function.
         delete vi->second;
         ++vi;
      }
      
      NSImportMap::iterator nsii = m_nsImports.begin();
      while( nsii != m_nsImports.end() )
      {
         // set the module to 0, so that we're not dec-reffed.
         delete nsii->second;         
         ++nsii;
      }
   }

   
   Dependency* getDep( const String& sourcemod, bool bIsUri, const String& symname, bool bSearchNS = false )
   {
      // has the symbolname a namespace translation?
      String remSymName;      
      Requirement* req = 0;
      
      length_t pos;
      if( bSearchNS && (pos = symname.rfind(".")) != String::npos )
      {
         // for sure, it has a namespace; but has a translation?
         String localNS = symname.subString(0,pos);
         NSImportMap::iterator nsi = m_nsImports.find( localNS );
         if( nsi != m_nsImports.end() )
         {
            // yep, we have an import namespace translator.
            req = nsi->second->m_req;
            remSymName = nsi->second->m_remNS + "." + symname.subString(pos+1);
         }
      }
      
      // if not found, or if we don't even want to search it, use the default
      if( req == 0 )
      {
        remSymName = symname;
        req = getReq( sourcemod, bIsUri );
      }
            
      Dependency* dep;
      Requirement::DepMap::iterator idep = req->m_deps.find( symname );
      if( idep != req->m_deps.end() )
      {
         dep = idep->second;
      }
      else
      {
         
         dep = new Private::Dependency( remSymName );
         req->m_deps[symname] = dep;
      }

      return dep;
   }
   
   
   Requirement* getReq( const String& sourcemod, bool bIsUri )
   {
      Requirement* req;
      ReqMap::iterator ireq = m_reqs.find( sourcemod );
      if( ireq != m_reqs.end() )
      {
         // already loaded?
         req = ireq->second;
         // it is legal to import symbols even from loaded modules.
      }
      else
      {
         req = new Requirement(sourcemod, false, bIsUri );
         m_reqs[sourcemod] = req;
      }
      return req;
   }
      
   /** Prepares the whole namespace import
    */
   bool addNSImport( const String& localNS, const String& remoteNS, Requirement* req )
   {
      // can't import a more than a single remote whole ns in a local ns
      NSImportMap::iterator iter = m_nsImports.find( localNS );
      if( iter != m_nsImports.end() )
      {
         return false;
      }
      
      m_nsImports[ localNS ] = new NSImport( remoteNS, req );
      return true;
   }
};

//=========================================================
// Main module class
//

Module::Module( const String& name ):
   m_name( name ),
   m_lastGCMark(0),
   m_bExportAll( false ),
   m_unloader( 0 )
{
   TRACE("Creating internal module '%s'", name.c_ize() );
   m_uri = "internal:" + name;
   _p = new Private;
}


Module::Module( const String& name, const String& uri ):
   m_name( name ),
   m_uri(uri),
   m_lastGCMark(0),
   m_bExportAll( false ),
   m_unloader( 0 )
{
   TRACE("Creating module '%s' from %s",
      name.c_ize(), uri.c_ize() );
   
   _p = new Private;
}


Module::~Module()
{
   TRACE("Deleting module %s", m_name.c_ize() );
   
   
   fassert2( m_unloader == 0, "A module cannot be destroyed with an active unloader!" );
   if( m_unloader != 0 )
   {      
      throw std::runtime_error( "A module cannot be destroyed with an active unloader!" );
   }

   // this is doing to do a bit of stuff; see ~Private()
   delete _p;   
   TRACE("Module '%s' deletion complete", m_name.c_ize() );
}


void Module::addStaticData( Class* cls, void* data )
{
   _p->m_staticdData.append( Item(cls, data) );
}


void Module::addAnonFunction( Function* f )
{
   // finally add to the function vecotr so that we can account it.
   String name = f->name();
   int count = 0;
   while( _p->m_functions.find( name ) != _p->m_functions.end() )
   {
      name = f->name();
      name.A("_").N(count++);
   }

   f->name(name);

   _p->m_functions[name] = f;
   f->module(this);

   // if this anonymous function was temporarily added as static data, we can remove it.
   if( ! _p->m_staticdData.empty()
         && _p->m_staticdData.at(_p->m_staticdData.length()-1).asInst() == f )
   {
      _p->m_staticdData.length( _p->m_staticdData.length()-1 );
   }

}


void Module::sendDynamicToGarbage()
{
   static Collector* coll = Engine::instance()->collector();

   ItemArray& ia = _p->m_staticdData;
   if( ia.empty() )
   {
      return;
   }

   // and get rid of the static data, if we have.
   length_t is = ia.length()-1;
   
   while( is < ia.length() )
   {
      Item& itm = ia[is];
      Class* handler = itm.asClass();
      if ( handler->typeID() != FLC_CLASS_ID_CLASS
         || handler->typeID() != FLC_CLASS_ID_FUNCTION )
      {
         FALCON_GC_STORE( coll, handler, itm.asInst() );
         ia.remove( is );
      }
      else
      {
         ++is;
      }
   }
}


GlobalSymbol* Module::addFunction( Function* f, bool bExport )
{
   //static Engine* eng = Engine::instance();

   Private::SymbolGuardMap& syms = _p->m_gSyms;
   if( syms.find(f->name()) != syms.end() )
   {
      return 0;
   }

   // add the symbol to the symbol table.
   GlobalSymbol* sym = new GlobalSymbol( f->name(), f );
   syms[f->name()] = Private::SymbolGuard(sym);

   // Eventually export it.
   if( bExport )
   {
      // by hypotesis, we can't have a double here, already checked on m_gSyms
      _p->m_gExports[f->name()] = sym;
   }

   // finally add to the function vecotr so that we can account it.
   _p->m_functions[f->name()] = f;
   f->module(this);

   return sym;
}


void Module::addFunction( GlobalSymbol* gsym, Function* f )
{
   //static Engine* eng = Engine::instance();

   // finally add to the function vecotr so that we can account it.
   _p->m_functions[f->name()] = f;
   f->module(this);

   if(gsym)
   {
      *gsym->value(0) = f;
   }
}


GlobalSymbol* Module::addFunction( const String &name, ext_func_t f, bool bExport )
{
   // check if the name is free.
   Private::SymbolGuardMap& syms = _p->m_gSyms;
   if( syms.find(name) != syms.end() )
   {
      return 0;
   }

   // ok, the name is free; add it
   Function* extfunc = new ExtFunc( name, f, this );
   return addFunction( extfunc, bExport );
}


GlobalSymbol* Module::addVariable( const String& name, bool bExport )
{
   return addVariable( name, Item(), bExport );
}


void Module::addClass( GlobalSymbol* gsym, Class* fc, bool )
{
   static Class* ccls = Engine::instance()->metaClass();

   // finally add to the function vecotr so that we can account it.
   _p->m_classes[fc->name()] = fc;
   fc->module(this);

   if(gsym)
   {
      gsym->value(0)->setUser( ccls, fc );
   }
}


GlobalSymbol* Module::addClass( Class* fc, bool, bool bExport )
{
   static Class* ccls = Engine::instance()->metaClass();

   Private::SymbolGuardMap& syms = _p->m_gSyms;
   if( syms.find(fc->name()) != syms.end() )
   {
      return 0;
   }

   // add a proper object in the global vector
   // add the symbol to the symbol table.
   GlobalSymbol* sym = new GlobalSymbol( fc->name(), Item(ccls, fc) );
   syms[fc->name()] = Private::SymbolGuard(sym);

   // Eventually export it.
   if( bExport )
   {
      // by hypotesis, we can't have a double here, already checked on m_gSyms
      _p->m_gExports[fc->name()] = sym;
   }

   // finally add to the function vecotr so that we can account it.
   _p->m_classes[fc->name()] = fc;
   fc->module(this);

   return sym;
}


void Module::addAnonClass( Class* cls )
{
   // finally add to the function vecotr so that we can account it.
   String name = cls->name();
   int count = 0;
   while( _p->m_classes.find( name ) != _p->m_classes.end() )
   {
      name = cls->name();
      name.A("_").N(count++);
   }

   cls->name(name);

   _p->m_classes[name] = cls;
   cls->module(this);

   // if this anonymous class was temporarily added as static data, we can remove it.
   if( ! _p->m_staticdData.empty()
         && _p->m_staticdData.at(_p->m_staticdData.length()-1).asInst() == cls )
   {
      _p->m_staticdData.length( _p->m_staticdData.length()-1 );
   }
}


GlobalSymbol* Module::addVariable( const String& name, const Item& value, bool bExport )
{
   GlobalSymbol* sym;
   
   // check if the name is free.
   Private::SymbolGuardMap& syms = _p->m_gSyms;
   Private::SymbolGuardMap::iterator pos = syms.find(name);
   if( pos != syms.end() )
   {
      sym = 0;
   }
   else
   {
      // add the symbol to the symbol table.
      sym = new GlobalSymbol( name, value );
      syms[name] = sym;
      if( bExport )
      {
         _p->m_gExports[name] = sym;
      }     
   }

   return sym;
}


Symbol* Module::getGlobal( const String& name ) const
{
   const Private::SymbolGuardMap& syms = _p->m_gSyms;
   Private::SymbolGuardMap::const_iterator iter = syms.find(name);

   if( iter == syms.end() )
   {
      return 0;
   }

   return iter->second.m_sym;
}


Function* Module::getFunction( const String& name ) const
{
   const Private::FunctionMap& funcs = _p->m_functions;
   Private::FunctionMap::const_iterator iter = funcs.find( name );

   if( iter == funcs.end() )
   {
      return 0;
   }

   return iter->second;
}


void Module::enumerateGlobals( SymbolEnumerator& rator ) const
{
   const Private::SymbolGuardMap& syms = _p->m_gSyms;
   Private::SymbolGuardMap::const_iterator iter = syms.begin();

   while( iter != syms.end() )
   {
      Symbol* sym = iter->second.m_sym;
      if( ! rator( *sym, ++iter == syms.end()) )
         break;
   }
}


void Module::enumerateExports( SymbolEnumerator& rator ) const
{
   if( m_bExportAll )
   {
      const Private::SymbolGuardMap& syms = _p->m_gSyms;   
      Private::SymbolGuardMap::const_iterator iter = syms.begin();

      while( iter != syms.end() )
      {
         Symbol* sym = iter->second.m_sym;
         // ignore "private" symbols
         if( sym->name().startsWith("_") || sym->type() != Symbol::t_global_symbol )
         {
            ++iter;
         }
         else
         {
            if( ! rator( *sym, ++iter == syms.end()) )
               break;
         }
      }
   }
   else
   { 
      const Private::GlobalsMap& syms = _p->m_gExports;
      Private::GlobalsMap::const_iterator iter = syms.begin();

      while( iter != syms.end() )
      {
         Symbol* sym = iter->second;
         // ignore "private" symbols
         if( sym->name().startsWith("_") || sym->type() != Symbol::t_global_symbol )
         {
            ++iter;
         }
         else
         {
            if( ! rator( *sym, ++iter == syms.end()) )
               break;
         }
      }
   }
}

   
bool Module::addLoad( const String& mod_name, bool bIsUri )
{
   // do we have the recor?
   Private::ReqMap::iterator ireq = _p->m_reqs.find( mod_name );
   if( ireq != _p->m_reqs.end() )
   {
      // already loaded?
      Private::Requirement* r = ireq->second;
      if( r->m_bIsLoad )
      {
         return false;
      }
      r->m_bIsLoad = true;
      return true;
   }

   // add a new requirement with load request
   Private::Requirement* r = new Private::Requirement( mod_name, true, bIsUri);
   _p->m_reqs[ mod_name ] = r;
   return true;
}


bool Module::addGenericImport( const String& source, bool bIsUri )
{
   Private::ReqMap::iterator pos = _p->m_reqs.find( source );
   Private::Requirement* req;
   if( pos != _p->m_reqs.end() )
   {
      // can we promote the module to generic import?
      req = pos->second;
      if( req->m_bIsGenericProvider )
      {
         // sorry, already promoted
         return false;
      }
   }
   else
   {
      // create the requirement now.
      req = new Private::Requirement( source, false, bIsUri );
      _p->m_reqs[source] = req;
   }
   
   // If we're here, we can grant promotion.
   req->m_bIsGenericProvider = true;
   _p->m_genericMods.push_back( req );
   return true;
}


UnknownSymbol* Module::addImportFrom( const String& localName, const String& remoteName,
                                        const String& source, bool bIsUri )
{
   // We can't be called if the symbol is alredy declared elsewhere.
   if( _p->m_gSyms.find( localName ) != _p->m_gSyms.end() )
   {
      return 0;
   }

   Private::Dependency* dep = _p->getDep( source, bIsUri, remoteName );
   UnknownSymbol* usym = new UnknownSymbol(localName);
      // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new Private::WaitingSym( usym ) );
   // ... and save the dependency.
   _p->m_gSyms[localName] = usym;
   
   return usym;
}


UnknownSymbol* Module::addImport( const String& name )
{
   // We can't be called if the symbol is alredy declared elsewhere.
   if( _p->m_gSyms.find( name ) != _p->m_gSyms.end() )
   {
      return 0;
   }

   // Get the special empty dependency for pure imports.
   Private::Dependency* dep = _p->getDep( "", false, name );
   
   UnknownSymbol* usym = new UnknownSymbol(name);
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new Private::WaitingSym( usym ) );
   // ... and save the dependency.
   _p->m_gSyms[name] = usym;

   return usym;
}


bool Module::addImplicitImport( UnknownSymbol* uks )
{
   // We can't be called if the symbol is alredy declared elsewhere.
   if( _p->m_gSyms.find( uks->name() ) != _p->m_gSyms.end() )
   {
      return false;
   }

   Private::Dependency* dep = _p->getDep( "", false, uks->name(), true );
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new Private::WaitingSym( uks ) );
   // ... and save the dependency.
   _p->m_gSyms[uks->name()] = uks;

   return true;
}


void Module::addImportRequest( Module::t_func_import_req func, const String& symName, const String& sourceMod, bool bModIsPath)
{   
   Private::Dependency* dep = _p->getDep( sourceMod, bModIsPath, symName );
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new Private::WaitingFunc( this, func ) );
}


Symbol* Module::addExport( const String& name, bool& bAlready )
{
   Symbol* sym = 0;
   // if we have export all or if this is already exported, return 0.
   Private::GlobalsMap::const_iterator eiter; 
   if( m_bExportAll || (( eiter = _p->m_gExports.find( name )) != _p->m_gExports.end()) )
   {
      sym = eiter->second;
      bAlready = true;
      return sym;
   }   
   
   // We can't be called if the symbol is alredy declared elsewhere.
   Private::SymbolGuardMap::iterator iter = _p->m_gSyms.find( name );
   if( iter != _p->m_gSyms.end() )
   {
      sym = iter->second.m_sym;
      
      // ... and save the dependency.
      _p->m_gExports[name] = sym;  
   }
   
   bAlready = false;   
   return sym;
}


void Module::addImportInheritance( Inheritance* inh )
{
   // Inheritances with dots are dependent on the given module.
   String ModName, inhName;
   inhName = inh->className();
   length_t pos = inhName.rfind(".");
   if( pos != String::npos )
   {
      ModName = inhName.subString(0,pos);
      inhName = inhName.subString(pos);
   }
   
   Private::Dependency* dep = _p->getDep( ModName, false, inhName );
   
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new Private::WaitingInherit( inh ) );
}


bool Module::passiveLink( ModSpace* ms )
{
   Private::ReqMap::iterator iter = _p->m_reqs.begin();
   while( iter != _p->m_reqs.end() )
   {
      Private::Requirement* req = iter->second;
      // loaded symbols from import...
      if( req->m_uri == "" )
      {
         Private::Requirement::DepMap::iterator iter = req->m_deps.begin();
         while( iter != req->m_deps.end() )
         {
            Private::Dependency* dep = iter->second;
            Module* mod;
            
            // first, try to resolve the symbol in the generic export list.
            Symbol* sym = findGenericallyExportedSymbol( dep->m_remoteName, mod );
               
            // better luck with the global exports
            if ( sym == 0 )
            {
               sym = ms->findExportedSymbol( dep->m_remoteName, mod );
            }
            
            if( sym == 0 )
            {
               ms->addLinkError( e_undef_sym, m_uri, 0, dep->m_remoteName );
            }
            else
            {
               // we're sending the module where this item is resolved, not the source.
               dep->resolved( this, sym );
               if( ! dep->m_errors.empty() )
               {
                  Private::Dependency::ErrorList::iterator erri = dep->m_errors.begin();
                  // If we have errors, transfer them to the module space.
                  while (erri != dep->m_errors.end() )
                  {
                     ms->addLinkError( *erri );
                  }
                  dep->clearErrors();
               }
            }
            ++iter;
         }
      }
      
      Module* mod = req->m_module;
      // if the module has been resolved, it won't be 0.
      if( mod != 0 )
      {
         Private::Requirement::DepMap::iterator iter = req->m_deps.begin();
         while( iter != req->m_deps.end() )
         {
            Private::Dependency* dep = iter->second;
            Symbol* sym = mod->getGlobal( dep->m_remoteName );
            if( sym == 0 )
            {
               ms->addLinkError( e_undef_sym, m_uri, 0, dep->m_remoteName );
            }
            else
            {
               // we're sending the module where this item is resolved, not the source.
               dep->resolved( this, sym );
               if( ! dep->m_errors.empty() )
               {
                  Private::Dependency::ErrorList::iterator erri = dep->m_errors.begin();
                  // If we have errors, transfer them to the module space.
                  while (erri != dep->m_errors.end() )
                  {
                     ms->addLinkError( *erri );
                  }
                  dep->clearErrors();
               }
            }
            ++iter;
         }
         
         // performs also the namespace forwarding
         Private::Requirement::NsImportMap::iterator nsiter = req->m_fullImport.begin();
         while( nsiter != req->m_fullImport.end() )
         {
            const String& remoteNS = nsiter->first;
            const String& localNS = nsiter->second;
            // search the symbols in the remote namespace.
            Private::SymbolGuardMap& globs = req->m_module->_p->m_gSyms;
            String nsPrefix = remoteNS + ".";
            Private::SymbolGuardMap::iterator gi = globs.lower_bound(nsPrefix);
            while( gi != globs.end() && gi->first.startsWith(nsPrefix) )
            {
               String localName = localNS + "." + gi->first.subString(nsPrefix.length());
               if( _p->m_gSyms.find(localName) == _p->m_gSyms.end())
               {
                  _p->m_gSyms[localName] = Private::SymbolGuard(gi->second.m_sym, false);
               }
               ++gi;
            }
            
            ++nsiter;
         }
      }
      ++iter;
   }
   
   return false;
}


Symbol* Module::findGenericallyExportedSymbol( const String& symname, Module*& declarer ) const
{
   Private::ReqList::const_iterator pos = _p->m_genericMods.begin();
   
   // found in?
   while( pos != _p->m_genericMods.end() )
   {
      Private::Requirement* req = *pos;
      if( req->m_module != 0 )
      {
         Symbol *sym = req->m_module->getGlobal( symname );
         if( sym != 0 )
         {
            declarer = req->m_module;
            return sym;
         }
      }
      
      ++pos;
   }
   
   return 0;
}


void Module::storeSourceClass( FalconClass* fcls, bool isObject, GlobalSymbol* gs )
{
   Class* cls = fcls;
   
   // The interactive compiler won't call us here if we have some undefined class,
   // as such, the construct can fail only if some class is not a falcon class.
   if( !fcls->construct() )
   {
      // did we fail to construct because we're incomplete?
      if( !fcls->missingParents() )
      {
         // so, we have to generate an hyper class out of our falcon-class
         // -- the hyperclass is also owning the FalconClass.
         cls = fcls->hyperConstruct();         
      }
   }

   if( gs == 0 )
   {
      addAnonClass( cls );
   }
   else
   {
      addClass( gs, cls, isObject );
   }
}


void Module::completeClass(FalconClass* fcls)
{                  
   // Completely resolved!
   if( !fcls->isPureFalcon() )
   {
      HyperClass* hcls = fcls->hyperConstruct();
      _p->m_classes[hcls->name()] = hcls;
      // anonymous classes cannot have a name in the global symbol table, so...
      Private::SymbolGuardMap::iterator pos = _p->m_gSyms.find( hcls->name() );
      if( pos != _p->m_gSyms.end() )
      {
         Item* value = pos->second.m_sym->value(0);
         value->setUser( value->asClass(), hcls );
      }
   }
}


bool Module::resolveStaticReqs( ModSpace* space )
{
   Private::ReqMap& reqs = _p->m_reqs;
   Private::ReqMap::iterator iter = reqs.begin();
   while( iter != reqs.end() )
   {
      Private::Requirement& req = *iter->second;
      
      // skip the special null requirement for implicit imports.      
      if( req.m_uri == "" )
      {
         ++iter;
         continue;
      }
      
      // already in?
      bool bLoad;
      Module* mod;
      if( (mod = space->findModule( req.m_uri, bLoad )) != 0 )
      {
         // need to promote?
         if( ! bLoad && req.m_bIsLoad )
         {
            space->promoteLoad( req.m_uri );
         }
         //anyhow add to us.
         req.m_module = mod;
         req.m_bIsDynamic = false;
         
      }
      else
      {
         // try to load
         ModLoader* loader = space->modLoader();
         if( req.m_bIsUri )
         {
            // shall throw on error.
            mod = loader->loadFile( req.m_uri, ModLoader::e_mt_none, true );                        
         }
         else
         {
            // shall throw on error.
            mod = loader->loadName( req.m_uri, ModLoader::e_mt_none );                        
         }
         if( mod == 0 )
         {
            // the only reason mod can be 0 here is that we didn't find it.
            throw new LinkError( ErrorParam( e_loaderror, __LINE__, SRC )
               .origin( ErrorParam::e_orig_linker )
               .extra( req.m_uri ));
         }
         // we don't really need to add it now, but...
         req.m_module = mod;
         req.m_bIsDynamic = false;
         
         // add it to the space -- which might fire other loads.
         space->addModule( mod, req.m_bIsLoad, false );
      }
      
      ++iter;
   }
   
   // TODO: Throw or return false on error?
   return true;
}


bool Module::resolveDynReqs( ModLoader* loader )
{
   Private::ReqMap& reqs = _p->m_reqs;
   Private::ReqMap::iterator iter = reqs.begin();
   while( iter != reqs.end() )
   {
      Private::Requirement& req = *iter->second;
      
      // skip the special null requirement for implicit imports.      
      if( req.m_uri == "" )
      {
         ++iter;
         continue;
      }
      
      Module* mod;
      if( req.m_bIsUri )
      {
         // shall throw on error.
         mod = loader->loadFile( req.m_uri, ModLoader::e_mt_none, true );                        
      }
      else
      {
         // shall throw on error.
         mod = loader->loadName( req.m_uri, ModLoader::e_mt_none );                        
      }
      // we don't really need to add it now, but...
      req.m_module = mod;
      req.m_bIsDynamic = true;
               
      ++iter;
   }
   
   // TODO: Throw or return false on error?
   return true;
}


void Module::unload()
{
   if( m_unloader != 0 )
   {
      DynUnloader* ul = m_unloader;
      m_unloader = 0;
      delete this;
      ul->unload();
   }
   else
   {
      delete this;
   }
}


bool Module::addImportFromWithNS( const String& localNS, const String& remoteName, 
            const String& modName, bool isFsPath )
{   
   Private::Requirement* req = _p->getReq( modName, isFsPath );
   
   // generic import from/in ns?
   if( remoteName == "" || remoteName == "*" )
   {
      // add a namespace requirement so that we know where to search for symbols
      if( localNS == "" )
      {
         if ( ! addGenericImport( modName, isFsPath ) )
         {
            return false;
         }
      }
      else if( ! _p->addNSImport( localNS, "", req ) )
      {
         return false; 
      }
      
      if ( remoteName == "*" )
      {
         req->m_fullImport[""] = localNS; // ok also with localNS == ""
      }
   }
   else
   {
      // is the source name composed?
      length_t posDot = remoteName.rfind( "." );
      if( posDot != String::npos )
      {
         String remPrefix = remoteName.subString( 0, posDot );
         String remDetail = remoteName.subString(posDot + 1);
         
         // import the whole thing in a namespace.
         if( remDetail == "*" )
         {
            // add a namespace requirement so that we know where to search for symbols      
            if( ! _p->addNSImport( localNS, remPrefix, req ) )
            {
               return false; 
            }
            
            req->m_fullImport[remPrefix] = localNS;
         }
         else
         {
            addImportFrom( localNS + "." + remDetail,  remoteName, modName, isFsPath );
         }
      }
      else
      {
         // we have just to add the symbol as-is
         addImportFrom( localNS + "." + remoteName,  remoteName, modName, isFsPath );
      }
   }
   
   return true;
}

}

/* end of module.cpp */
