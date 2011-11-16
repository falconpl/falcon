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
#include <falcon/symbol.h>
#include <falcon/extfunc.h>
#include <falcon/item.h>
#include <falcon/modspace.h>
#include <falcon/modloader.h>
#include <falcon/inheritance.h>
#include <falcon/requirement.h>
#include <falcon/error.h>
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/dynunloader.h>

#include <falcon/errors/genericerror.h>
#include <falcon/errors/linkerror.h>

#include <stdexcept>
#include <map>
#include <deque>
#include <list>

#include "module_private.h"

namespace Falcon {
      
Error* Module::Private::WaitingFunc::onSymbolLoaded( Module* mod, Symbol* sym )
{
   return m_func( m_requester, mod, sym );
}


Error* Module::Private::WaitingInherit::onSymbolLoaded( Module* mod, Symbol* sym )
{
   Item* value;
   if( (value = sym->value( 0 )) == 0 || ! value->isClass() )
   {
      // the symbol is not global?            
      return new CodeError( ErrorParam( e_inv_inherit ) 
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
   

Error* Module::Private::WaitingRequirement::onSymbolLoaded( Module* mod, Symbol* sym )
{
   return m_cr->resolve( mod, sym );
}


Module::Private::Dependency::~Dependency()
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

      
void Module::Private::Dependency::clearErrors() {
   ErrorList::iterator eli = m_errors.begin();
   while( eli != m_errors.end() )
   {
      (*eli)->decref();
      ++eli;
   }
   m_errors.clear();
}
      
void Module::Private::Dependency::resolved( Module* mod, Symbol* sym )
{
   m_resolvedSymbol = sym;
   m_resolvedModule = mod;
   
   if( m_symbol != 0 )
   {
      m_symbol->define( Symbol::e_st_global, sym->id() );
      m_symbol->defaultValue( sym->defaultValue() );
   }
   
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


Error* Module::Private::Dependency::resolveOnModSpace( ModSpace* ms, const String& uri, int line )
{
   fassert( ! m_waiting.empty() );
   
   if( m_resolvedSymbol != 0 )
   {
      m_waiting.back()->onSymbolLoaded( m_resolvedModule, m_resolvedSymbol );
   }
   else 
   {
      const String& symName = m_remoteName;
      Module* declarer;
      Symbol* sym = ms->findExportedSymbol( symName, declarer );
      if( sym != 0 )
      {
         resolved( declarer, sym );               
      }
      else {
         return new LinkError( ErrorParam(e_undef_sym, line, uri )
            .extra( symName )
            );

      }
   }

   // do we have some errors?
   Module::Private::Dependency::ErrorList::iterator ierr = m_errors.begin();
   Error* err = 0;
   while( ierr != m_errors.end() ) {
      if( err != 0 )
      {
         err->appendSubError( *ierr );
      }
      else
      {
         err = *ierr;
      }
      ++ierr;
   }
   clearErrors();
   
   return err;
}

Module::Private::Request::~Request()
{
   DepMap::iterator dep_i = m_deps.begin();
   while( dep_i != m_deps.end() )
   {
      delete dep_i->second;
      ++dep_i;
   }
}
      
      
Module::Private::~Private()
{
   // We can destroy the globals, as we're always responsible for that...
   GlobalsMap::iterator iter = m_gSyms.begin();
   while( iter != m_gSyms.end() )
   {
      Symbol* sym = iter->second;
      delete sym;
      ++iter;
   }

   // and get rid of the static data, if we have.
   StaticDataList::iterator sditer = m_staticData.begin();
   while( sditer != m_staticData.end() )
   {
      Item& itm = *sditer;      
      
      Class* cls; 
      void* inst;
      if( itm.asClassInst( cls, inst ) )
      {
         cls->dispose( inst );
      }
      
      ++sditer;
   }

   // destroy reqs and deps
   ReqMap::iterator req_i = m_reqs.begin();
   while( req_i != m_reqs.end() )
   {
      delete req_i->second;
      ++req_i;
   }

   NSImportMap::iterator nsii = m_nsImports.begin();
   while( nsii != m_nsImports.end() )
   {
      // set the module to 0, so that we're not dec-reffed.
      delete nsii->second;         
      ++nsii;
   }
}


Module::Private::Dependency* Module::Private::getDep( 
      const String& sourcemod, bool bIsUri, const String& symname, bool bSearchNS )
{
   // has the symbolname a namespace translation?
   String remSymName;      
   Request* req = 0;

   length_t pos;
   if( bSearchNS && (pos = symname.rfind( '.')) != String::npos )
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
   Request::DepMap::iterator idep = req->m_deps.find( symname );
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
   

Module::Private::Dependency* Module::Private::findDep( const String& sourcemod, const String& symname ) const
{
   ReqMap::const_iterator iter = m_reqs.find( sourcemod );
   if( iter == m_reqs.end() )
   {
      return 0;
   }
   
   Request::DepMap::const_iterator diter = iter->second->m_deps.find( symname );
   if( diter == iter->second->m_deps.end() )
   {
      return 0;
   }
   
   return diter->second;
}

void Module::Private::removeDep( const String& sourcemod, const String& symname, bool bClearReq )
{
   ReqMap::iterator iter = m_reqs.find( sourcemod );
   if( iter == m_reqs.end() )
   {
      return;
   }
   
   Request::DepMap& deps = iter->second->m_deps;
   Request::DepMap::iterator diter = deps.find( symname );
   if( diter == deps.end() )
   {
      return;
   }
   
   deps.erase( diter );
   if( bClearReq && deps.empty() )
   {
      m_reqs.erase( iter );
   }   
}
   
   
Module::Private::Request* Module::Private::getReq( const String& sourcemod, bool bIsUri )
{
   Request* req;
   ReqMap::iterator ireq = m_reqs.find( sourcemod );
   if( ireq != m_reqs.end() )
   {
      // already loaded?
      req = ireq->second;
      // it is legal to import symbols even from loaded modules.
   }
   else
   {
      req = new Request(sourcemod, e_lm_import_public , bIsUri );
      m_reqs[sourcemod] = req;
   }
   return req;
}


bool Module::Private::addNSImport( const String& localNS, const String& remoteNS, Request* req )
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


Module::Private::Dependency* Module::Private::addRequirement( Requirement * cr )
{
   // Inheritances with dots are dependent on the given module.
   String ModName, crName;
   crName = cr->name();
   length_t pos = crName.rfind( '.' );
   if( pos != String::npos )
   {
      ModName = crName.subString(0,pos);
      crName = crName.subString(pos);
   }
   
   Dependency* dep = getDep( ModName, false, crName );
   
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new WaitingRequirement( cr ) );
   return dep;
}

//=========================================================
// Main module class
//

Module::Module( const String& name ):
   m_modSpace(0),
   m_name( name ),
   m_lastGCMark(0),
   m_bExportAll( false ),
   m_unloader( 0 ),
   m_bMain( false ),
   m_anonFuncs(0),
   m_anonClasses(0)
{
   TRACE("Creating internal module '%s'", name.c_ize() );
   m_uri = "internal:" + name;
   _p = new Private;
}


Module::Module( const String& name, const String& uri ):
   m_modSpace(0),
   m_name( name ),
   m_uri(uri),
   m_lastGCMark(0),
   m_bExportAll( false ),
   m_unloader( 0 ),
   m_bMain( false ),
   m_anonFuncs(0),
   m_anonClasses(0)
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


void Module::gcMark( uint32 mark )
{
   if( m_lastGCMark != mark )
   {
      m_lastGCMark = mark;
      if ( m_modSpace != 0 )
      {
         m_modSpace->gcMark( mark );
      }
      
      Private::StaticDataList::iterator sditer = _p->m_staticData.begin();
      while( sditer != _p->m_staticData.end() )
      {
         sditer->gcMark( mark );
         ++sditer;
      }
      
   }
}


Item* Module::addStaticData( Class* cls, void* data )
{
   _p->m_staticData.push_back( Item(cls, data) );
   return &_p->m_staticData.back();
}


void Module::addAnonFunction( Function* f )
{
   // finally add to the function vecotr so that we can account it.
   String name;
   do
   {
      name = "lambda#";
      name.N(m_anonFuncs++);
   } while( _p->m_functions.find( name ) != _p->m_functions.end() );

   f->name(name);
   f->module(this);

   _p->m_functions[name] = f;
}


Symbol* Module::addFunction( Function* f, bool bExport )
{
   //static Engine* eng = Engine::instance();

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(f->name()) != syms.end() )
   {
      return 0;
   }

   // add the symbol to the symbol table.
   Symbol* sym = new Symbol( f->name(), Symbol::e_st_global );
   syms[f->name()] = sym;

   // Eventually export it.
   if( bExport )
   {
      // by hypotesis, we can't have a double here, already checked on m_gSyms
      _p->m_gExports[f->name()] = sym;
   }

   // finally add to the function vecotr so that we can account it.
   _p->m_functions[f->name()] = f;
   f->module(this);

   sym->defaultValue(addDefaultValue( f ));
   return sym;
}


void Module::addFunction( Symbol* gsym, Function* f )
{
   //static Engine* eng = Engine::instance();

   // finally add to the function vecotr so that we can account it.
   _p->m_functions[f->name()] = f;
   f->module(this);
   
   if(gsym)
   {
      Item* ival = addDefaultValue(f);
      gsym->defaultValue(ival);
   }
}


Symbol* Module::addFunction( const String &name, ext_func_t f, bool bExport )
{
   // check if the name is free.
   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(name) != syms.end() )
   {
      return 0;
   }

   // ok, the name is free; add it
   Function* extfunc = new ExtFunc( name, f, this );
   return addFunction( extfunc, bExport );
}


Symbol* Module::addVariable( const String& name, bool bExport )
{
   return addVariable( name, Item(), bExport );
}


void Module::addClass( Symbol* gsym, Class* fc, bool )
{
   static Class* ccls = Engine::instance()->metaClass();

   // finally add to the function vecotr so that we can account it.
   _p->m_classes[fc->name()] = fc;
   fc->module(this);

   if(gsym)
   {
      gsym->defaultValue( addStaticData( ccls, fc ) );
   }
   
}


Symbol* Module::addClass( Class* fc, bool, bool bExport )
{
   static Class* ccls = Engine::instance()->metaClass();

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(fc->name()) != syms.end() )
   {
      return 0;
   }

   // add a proper object in the global vector
   // add the symbol to the symbol table.
   Symbol* sym = new Symbol( fc->name(), Symbol::e_st_global );
   syms[fc->name()] = sym;

   // Eventually export it.
   if( bExport )
   {
      // by hypotesis, we can't have a double here, already checked on m_gSyms
      _p->m_gExports[fc->name()] = sym;
   }

   // finally add to the function vecotr so that we can account it.
   _p->m_classes[fc->name()] = fc;
   fc->module(this);
   sym->defaultValue( addStaticData( ccls, fc ) );

   return sym;
}


void Module::addAnonClass( Class* cls )
{
   // finally add to the function vecotr so that we can account it.
   String name;
   do
   {
      name = "class#";
      name.N(m_anonClasses++);
   } while ( _p->m_classes.find( name ) != _p->m_classes.end() );

   cls->name(name);

   _p->m_classes[name] = cls;
   cls->module(this);
}


Symbol* Module::addVariable( const String& name, const Item& value, bool bExport )
{
   Symbol* sym;
   
   // check if the name is free.
   Private::GlobalsMap& syms = _p->m_gSyms;
   Private::GlobalsMap::iterator pos = syms.find(name);
   if( pos != syms.end() )
   {
      sym = 0;
   }
   else
   {
      // add the symbol to the symbol table.
      sym = new Symbol( name, Symbol::e_st_global );
      syms[name] = sym;
      sym->defaultValue( addDefaultValue(value) );
      if( bExport )
      {
         _p->m_gExports[name] = sym;
      }     
   }

   return sym;
}


Symbol* Module::getGlobal( const String& name ) const
{
   const Private::GlobalsMap& syms = _p->m_gSyms;
   Private::GlobalsMap::const_iterator iter = syms.find(name);

   if( iter == syms.end() )
   {
      return 0;
   }

   return iter->second;
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
   const Private::GlobalsMap& syms = _p->m_gSyms;
   Private::GlobalsMap::const_iterator iter = syms.begin();

   while( iter != syms.end() )
   {
      Symbol* sym = iter->second;
      if( ! rator( *sym, ++iter == syms.end()) )
         break;
   }
}


void Module::enumerateExports( SymbolEnumerator& rator ) const
{
   if( m_bExportAll )
   {
      const Private::GlobalsMap& syms = _p->m_gSyms;   
      Private::GlobalsMap::const_iterator iter = syms.begin();

      while( iter != syms.end() )
      {
         Symbol* sym = iter->second;
         // ignore "private" symbols
         if( sym->name().startsWith("_") || sym->type() != Symbol::e_st_global )
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
         if( sym->name().startsWith("_") || sym->type() != Symbol::e_st_global )
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
      Private::Request* r = ireq->second;
      if( r->m_loadMode == e_lm_load )
      {
         return false;
      }
      r->m_loadMode = e_lm_load;
      return true;
   }

   // add a new requirement with load request
   Private::Request* r = new Private::Request( mod_name, e_lm_load, bIsUri);
   _p->m_reqs[ mod_name ] = r;
   return true;
}


bool Module::addGenericImport( const String& source, bool bIsUri )
{
   Private::ReqMap::iterator pos = _p->m_reqs.find( source );
   Private::Request* req;
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
      req = new Private::Request( source, e_lm_load, bIsUri );
      _p->m_reqs[source] = req;
   }
   
   // If we're here, we can grant promotion.
   req->m_bIsGenericProvider = true;
   _p->m_genericMods.push_back( req );
   return true;
}


Symbol* Module::addImportFrom( const String& localName, const String& remoteName,
                                        const String& source, bool bIsUri )
{
   // We can't be called if the symbol is alredy declared elsewhere.
   if( _p->m_gSyms.find( localName ) != _p->m_gSyms.end() )
   {
      return 0;
   }

   Private::Dependency* dep = _p->getDep( source, bIsUri, remoteName );
   Symbol* usym = new Symbol( localName, Symbol::e_st_extern );
   dep->m_symbol = usym;
   // ... and save the dependency.
   _p->m_gSyms[localName] = usym;
   
   return usym;
}


Symbol* Module::addImport( const String& name )
{
   // We can't be called if the symbol is alredy declared elsewhere.
   if( _p->m_gSyms.find( name ) != _p->m_gSyms.end() )
   {
      return 0;
   }

   // Get the special empty dependency for pure imports.
   Private::Dependency* dep = _p->getDep( "", false, name );
   
   Symbol* usym = new Symbol( name, Symbol::e_st_extern );
   dep->m_symbol = usym;   
   // ... and save the dependency.
   _p->m_gSyms[name] = usym;
   
   return usym;
}


Symbol* Module::addImplicitImport( const String& name )
{
   // We can't be called if the symbol is alredy declared elsewhere.
   if( _p->m_gSyms.find( name ) != _p->m_gSyms.end() )
   {
      return 0;
   }

   Private::Dependency* dep = _p->getDep( "", false, name, true );
   // Record the fact that we have to save transform an unknown symbol...
   Symbol* uks = new Symbol( name, Symbol::e_st_extern );
   dep->m_symbol = uks;
   // ... and save the dependency.
   _p->m_gSyms[name] = uks;

   return uks;
}


void Module::addImportRequest( Module::t_func_import_req func, const String& symName, const String& sourceMod, bool bModIsPath)
{   
   Private::Dependency* dep = _p->getDep( sourceMod, bModIsPath, symName );
   // here we have no symbol to save.
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
   Private::GlobalsMap::iterator iter = _p->m_gSyms.find( name );
   if( iter != _p->m_gSyms.end() )
   {
      sym = iter->second;
      _p->m_gExports[name] = sym;  
   }
   
   bAlready = false;   
   return sym;
}


void Module::addImportInheritance( Inheritance* inh )
{
   // we don't care if the symbol already exits; the mehtod would just return 0.
   Symbol* imported = addImplicitImport( inh->className() );
   if( imported == 0 )
   {
      // if addImplicitImport returns 0, then the symbol MUST be a global.
      imported = getGlobal( inh->className() );
      fassert( imported != 0 );
   }
   
   // Inheritances with dots are dependent on the given module.
   String ModName, inhName;
   inhName = inh->className();
   length_t pos = inhName.rfind( '.');
   if( pos != String::npos )
   {
      ModName = inhName.subString(0,pos);
      inhName = inhName.subString(pos);
   }
   
   Private::Dependency* dep = _p->getDep( ModName, false, inhName );
   dep->m_symbol = imported;
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new Private::WaitingInherit( inh ) );
}



void Module::addPendingInheritance( Inheritance* inh )
{
   // we don't care if the symbol already exits; the mehtod would just return 0.
   addImplicitImport( inh->className() );
   _p->m_pendingInh.push_back( inh );
}


bool Module::checkPendingInheritance(const String& symName, Class* parent)
{
   bool bFound = false;
   
   Private::InheritanceList::iterator iter = _p->m_pendingInh.begin();
   while( _p->m_pendingInh.end() != iter )
   {
      Inheritance* inh = *iter;
      if( inh->className() == symName )
      {
         bFound = true;
         if( parent == 0 ) 
         {
            break;
         }
         
         inh->parent( parent );
         iter = _p->m_pendingInh.erase( iter );
         if( parent->isFalconClass() )
         {
            // then, see if we can link it.
            FalconClass* falcls = static_cast<FalconClass*>(parent);
            if( falcls->missingParents() == 0 )
            {
               completeClass( falcls );
            }
         }
      }
      else
      {
         ++iter;
      }
   }
   
   return bFound;
}


void Module::commitPendingInheritance()
{   
   Private::InheritanceList::iterator iter = _p->m_pendingInh.begin();
   while( _p->m_pendingInh.end() != iter )
   {
      Inheritance* inh = *iter;
      addImportInheritance( inh );
      ++iter;
   }
    _p->m_pendingInh.clear();
}


void Module::addRequirement( Requirement* cr )
{
   _p->addRequirement(cr);
}


Error* Module::addRequirementAndResolve( Requirement* cr )
{
   Private::Dependency* dep = _p->addRequirement(cr);
   
   if( m_modSpace != 0 ) 
   {
      return dep->resolveOnModSpace( m_modSpace, uri(), cr->sourceRef().line() );
   }
   else
   {
      return new LinkError( ErrorParam(e_undef_sym, cr->sourceRef().line(), uri() )
            .extra( cr->name() )
            );
   }
}


void Module::storeSourceClass( FalconClass* fcls, bool isObject, Symbol* gs )
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
   if( !fcls->construct() )
   {
      // was not a full falcon class; we must change it into an hyper class.
      HyperClass* hcls = fcls->hyperConstruct();
      _p->m_classes[hcls->name()] = hcls;
      // anonymous classes cannot have a name in the global symbol table, so...
      Private::GlobalsMap::iterator pos = _p->m_gSyms.find( hcls->name() );
      if( pos != _p->m_gSyms.end() )
      {
         Item* value = pos->second->defaultValue();
         value->setUser( value->asClass(), hcls );
      }
   }
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
   Private::Request* req = _p->getReq( modName, isFsPath );
   
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
      length_t posDot = remoteName.rfind( '.' );
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
            return addImportFrom( localNS + "." + remDetail,  remoteName, modName, isFsPath ) != 0;
         }
      }
      else
      {
         // we have just to add the symbol as-is
         return addImportFrom( localNS + "." + remoteName,  remoteName, modName, isFsPath ) != 0;
      }
   }
   
   return true;
}



void Module::forwardNS( Module* mod, const String& remoteNS, const String& localNS )
{
   Private::GlobalsMap& globs = mod->_p->m_gSyms;
   
   // search the symbols in the remote namespace.
   String nsPrefix = remoteNS + ".";
   Private::GlobalsMap::iterator gi = globs.lower_bound(nsPrefix);
   while( gi != globs.end() && gi->first.startsWith(nsPrefix) )
   {
      String localName = localNS + "." + gi->first.subString(nsPrefix.length());
      if( _p->m_gSyms.find(localName) == _p->m_gSyms.end())
      {
         _p->m_gSyms[localName] = new Symbol(*gi->second);
         //TODO: Reference the udnerlying VM variable
      }
      ++gi;
   }
}


bool Module::anyImportFrom( const String& path, bool isFsPath, const String& symName,
      const String& nsName, bool bIsNS )
{
   if( nsName != "" )
   {
      if( bIsNS )
      {        
         return addImportFromWithNS( nsName, symName, path, isFsPath );
      }
      else
      {
         // it's an as -- and a pure one, as the parser removes possible "as" errors.
         return addImportFrom( nsName, symName, path, isFsPath ) != 0;
      }
   }
   else
   {
      if( symName == "" )
      {
         return addGenericImport( path, isFsPath );
      }
      else
      {
         if( symName.endsWith("*") )
         {
            // fake "import a.b.c.* from Module in a.b.c
            return addImportFromWithNS( 
                        symName.length() > 2 ? 
                                 symName.subString(0, symName.length()-2) : "", 
                        symName, path, isFsPath );
         }
         else
         {
            addImportFrom( symName, symName, path, isFsPath );
         }
      }
   }  
   
   return true;
}



Item* Module::addDefaultValue()
{
   _p->m_staticData.push_back(Item());
   return &_p->m_staticData.back();
}

Item* Module::addDefaultValue( const Item& src )
{
   _p->m_staticData.push_back( src );
   return &_p->m_staticData.back();
}

}

/* end of module.cpp */
