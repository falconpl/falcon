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
#include <falcon/modrequest.h>
#include <falcon/importdef.h>
#include <falcon/requirement.h>
#include <falcon/error.h>
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/dynunloader.h>

#include <falcon/errors/codeerror.h>
#include <falcon/errors/genericerror.h>
#include <falcon/errors/linkerror.h>
#include <falcon/classes/classrequirement.h>

#include <stdexcept>
#include <map>
#include <deque>
#include <list>
#include <algorithm>

#include "module_private.h"

namespace Falcon 
{

class Module::FuncRequirement: public Requirement
{
public:
   FuncRequirement( const String& symName, t_func_import_req& cbFunc ):
      Requirement( symName ),
      m_cbFunc( cbFunc )
   {}

   virtual ~FuncRequirement() {}

   virtual void onResolved( const Module* source, const Symbol* srcSym, Module* tgt, Symbol* )
   {
      Error* err = m_cbFunc( tgt, source, srcSym );
      if( err != 0 ) throw err;
   }
   
   // This applies only to native modules, which doesn't store requirementes.
   virtual Class* cls() const { return 0; }
   
private:
   t_func_import_req m_cbFunc;   
};
   
   
Error* Module::Private::Dependency::onResolved( Module* parentMod, Module* mod, Symbol* sym )
{
   Error* res = 0;
   m_resSymbol = sym;
   bool firstError = true;
   
   Private::Dependency::WaitingList::iterator iter = m_waitings.begin();
   while( m_waitings.end() != iter )
   {
      try 
      {
         Requirement* req = *iter;
         req->onResolved( mod, sym, parentMod, m_symbol );
      }
      catch( Error * e )
      {
         if( res == 0 )
         {
            res = e;
         }
         else
         {
            if( firstError )
            {
               firstError = false;
               Error* temp = res;
               res = new LinkError( ErrorParam( e_link_error, 0, parentMod->uri())
                  .extra( "Errors during symbol resolution")
                  );
               res->appendSubError(temp);
               temp->decref();
            }
            res->appendSubError(e);
            e->decref();            
         }
      }
      ++iter;
   }
   
   return res;
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

   // destroy reqs and deps
   DepList::iterator req_i = m_deplist.begin();
   while( req_i != m_deplist.end() )
   {
      delete *req_i;
      ++req_i;
   }
   
   ImportDefList::iterator id_i = m_importDefs.begin();
   while( id_i != m_importDefs.end() )
   {
      delete *id_i;
      ++id_i;
   }
      
   ReqList::iterator rl_i = m_mrlist.begin();
   while( rl_i != m_mrlist.end() )
   {
      delete *rl_i;
      ++rl_i;
   }
   
   NSImportList::iterator nsi = m_nsimports.begin();
   while( nsi != m_nsimports.end() )
   {
      Private::NSImport* ns = *nsi;
      delete ns;
      ++nsi;
   }
}


//=========================================================
// Main module class
//

Module::Module():
   m_modSpace(0),
   m_name( "$noname" ),
   m_lastGCMark(0),
   m_bExportAll( false ),
   m_unloader( 0 ),
   m_bMain( false ),
   m_anonFuncs(0),
   m_anonClasses(0),
   m_mainFunc(0),
   m_bNative( false )
{
   TRACE("Creating internal module '%s'", m_name.c_ize() );
   m_uri = "";
   _p = new Private;
}


Module::Module( const String& name, bool bNative ):
   m_modSpace(0),
   m_name( name ),
   m_lastGCMark(0),
   m_bExportAll( false ),
   m_unloader( 0 ),
   m_bMain( false ),
   m_anonFuncs(0),
   m_anonClasses(0),
   m_mainFunc(0),
   m_bNative( bNative )
{
   TRACE("Creating internal module '%s'", name.c_ize() );
   m_uri = "";
   _p = new Private;
}


Module::Module( const String& name, const String& uri, bool bNative ):
   m_modSpace(0),
   m_name( name ),
   m_uri(uri),
   m_lastGCMark(0),
   m_bExportAll( false ),
   m_unloader( 0 ),
   m_bMain( false ),
   m_anonFuncs(0),
   m_anonClasses(0),
   m_mainFunc(0),
   m_bNative( bNative )
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
      
      //TODO: Mark other modules and data in symbols      
   }
}


void Module::addAnonFunction( Function* f )
{
   // finally add to the function vecotr so that we can account it.
   String name;
   do
   {
      name = f->isEta() ? "eta#" : "lambda#";
      name.N(m_anonFuncs++);
   } while( _p->m_functions.find( name ) != _p->m_functions.end() );

   f->name(name);
   f->module(this);

   _p->m_functions[name] = f;
   
   // by definition, an anonymous function cannot cover forward refs
}


Symbol* Module::addFunction( Function* f, bool bExport )
{
   static Class* symClass = Engine::instance()->symbolClass();
   static Collector* coll = Engine::instance()->collector();

   //static Engine* eng = Engine::instance();

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(f->name()) != syms.end() )
   {
      return 0;
   }
   
   // add to the function vecotr so that we can account it.
   _p->m_functions[f->name()] = f;
   f->module(this);

   // add the symbol to the symbol table.
   Symbol* sym = new Symbol( f->name(), Symbol::e_st_global, 0, f->declaredAt() );
   FALCON_GC_STORE( coll, symClass, sym );
   sym->defaultValue(f);
   syms[f->name()] = sym;

   // Eventually export it.
   if( bExport )
   {
      // by hypotesis, we can't have a double here, already checked on m_gSyms
      _p->m_gExports[f->name()] = sym;
   }
   
   // see if this covers a forward declaration.
   checkWaitingFwdDef( sym );

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
      gsym->defaultValue(f);
   }
   
   // see if this covers a forward declaration.
   checkWaitingFwdDef( gsym );
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
      gsym->defaultValue( Item( ccls, fc ) );
      // see if this covers a forward declaration.
      checkWaitingFwdDef( gsym );
   }
}


Symbol* Module::addClass( Class* fc, bool, bool bExport )
{
   static Class* ccls = Engine::instance()->metaClass();
   static Class* csym = Engine::instance()->symbolClass();
   static Collector* coll = Engine::instance()->collector();
   

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(fc->name()) != syms.end() )
   {
      return 0;
   }

   // add a proper object in the global vector
   // add the symbol to the symbol table.
   Symbol* sym = new Symbol( fc->name(), Symbol::e_st_global, 0, fc->declaredAt() );
   FALCON_GC_STORE( coll, csym, sym );
   
   sym->defaultValue(Item(ccls, fc));
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
   // see if this covers a forward declaration.
   checkWaitingFwdDef( sym );

   return sym;
}

Class* Module::getClass( const String& name ) const
{
   Private::ClassMap::const_iterator iter = _p->m_classes.find(name);
   if( iter != _p->m_classes.end() )
   {
      return iter->second;
   }
   
   return 0;
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
   static Class* symClass = Engine::instance()->symbolClass();
   static Collector* coll = Engine::instance()->collector();

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
      sym = new Symbol( name, Symbol::e_st_global, 0, 0);
      FALCON_GC_STORE( coll, symClass, sym );
      sym->defaultValue(value);
      syms[name] = sym;
      if( bExport )
      {
         _p->m_gExports[name] = sym;
      }     
   }

   checkWaitingFwdDef( sym );
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


Error* Module::addModuleRequirement( ImportDef* def, ModRequest*& req )
{
   // first, check if there is a clash with already defined symbols.
   int symcount = def->symbolCount();
   for( int i = 0; i < symcount; ++ i )
   {
      String name;      
      def->targetSymbol(i, name );
      if( name.size() == 0 ) continue; // a bit defensive.
      
      if( name.getCharAt( name.length() -1 ) !=  '*' )
      {
         // it's a real symbol.
         if ( _p->m_gSyms.find( name ) != _p->m_gSyms.end() )
         {
            return new CodeError( ErrorParam( e_import_already, def->sr().line(), this->uri() ) 
               .origin( ErrorParam::e_orig_compiler )
               .extra(name) );
         }
      }
   }
   
   Private::ReqMap::iterator pos = _p->m_mrmap.find( def->sourceModule() );
   if( pos != _p->m_mrmap.end() )
   {
      req = pos->second;
      // prevent double load requests -- or redefining already known modules.
      if( def->isLoad() && req->isLoad() )
      {
         return new CodeError( ErrorParam( e_load_already, def->sr().line(), this->uri() )
            .origin( ErrorParam::e_orig_compiler )
            .extra( def->sourceModule() ) );
      }
      
      // update load status.
      if( def->isLoad() )
      {
         req->promoteLoad();
      }
      
      // update logical name into physical if there is a clash.
      // i.e. load test and load "test" will have "test" to prevail.
      if( def->isUri() )
      {
         req->isUri(true);
      }      
   }
   else
   {
      // create a new entry
      req = new ModRequest( def->sourceModule(), def->isUri(), def->isLoad() );
      _p->m_mrmap[def->sourceModule()] = req;
      _p->m_mrlist.push_back( req );
   }

   def->modReq( req );
   req->addImportDef( def );
   return 0;
}


bool Module::removeModuleRequirement( ImportDef* def )
{
   bool found = false;
   
   Private::ReqMap::iterator pos = _p->m_mrmap.find( def->sourceModule() );
   if( pos != _p->m_mrmap.end() )
   { 
      found = true;
      
      // search the def backward 
      // -- (usually, we remove the last def because something went wrong)
      ModRequest* req = pos->second;
      req->removeImportDef( def );
      if( req->importDefCount() == 0 )
      {
         // we don't have any reason to depend from this module anymore 
         _p->m_mrmap.erase(pos);
         
         // remove also from the generic mods, in case it was a generic provider
         Private::ReqList::iterator reqi = 
               std::find( _p->m_genericMods.begin(), _p->m_genericMods.end(), req );
         if( reqi != _p->m_genericMods.end() )
         {
            _p->m_genericMods.erase( reqi );
         }
         
         // And anyhow from the global list.
         reqi = std::find( _p->m_mrlist.begin(), _p->m_mrlist.end(), req );
         if( reqi != _p->m_mrlist.end() )
         {
            _p->m_genericMods.erase( reqi );
         }
         
         // we own the request.
         delete req;
      }
   }
   
   return found;
}


Error* Module::addImport( ImportDef* def )
{
   static Class* symClass = Engine::instance()->symbolClass();
   static Collector* coll = Engine::instance()->collector();
   
   ModRequest* req;
   Error* error = addModuleRequirement( def, req );
   if( error != 0 )
   {
      return error;
   }
      
   // check that all the symbols are locally undefined.
   int symcount = def->symbolCount();
      
   // ok we can proceed -- record all the symbols as externs.
   for( int i = 0; i < symcount; ++ i )
   {
      String name;      
      def->targetSymbol( i, name );
      if( name.size() == 0 ) continue; // a bit defensive.
      
      if( name.getCharAt( name.length() -1 ) != '*' )
      {         
         Symbol* newsym = new Symbol( name, Symbol::e_st_extern, 0, def->sr().line());
         FALCON_GC_STORE( coll, symClass, newsym );
         _p->m_gSyms[name] = newsym;
         
         // and add a dependency.
         Private::Dependency* dep = new Private::Dependency( newsym, def );         
         
         dep->m_sourceName = def->sourceSymbol( i );
         
         _p->m_depsBySymbol[name] = dep;
         _p->m_deplist.push_back( dep );
      }
      else if( def->sourceModule().size() != 0 )
      {
         // get the from part.
         String from;
         name = def->sourceSymbol( i );
         if ( name.length() > 2 )
         {
            from = name.subString(0, name.length()-2);
         }
         
         // we should just have added it in addModuleRequirement
         _p->m_nsimports.push_back( new Private::NSImport( def, from, def->target() ) );
      }
   }
   
   // save the definition
   _p->m_importDefs.push_back( def );
   
   // eventually, save the module as a generic provider.
   if( def->isGeneric() )
   {
      _p->m_genericMods.push_back( req );
   }
   
   return 0;
}


void Module::removeImport( ImportDef* def )
{
   removeModuleRequirement( def);
      
   // We know that all the symbols in this importdef were defined.
   int symcount = def->symbolCount();      
   for( int i = 0; i < symcount; ++ i )
   {
      String name;      
      def->targetSymbol( i, name );
      
      Private::GlobalsMap::iterator gep = _p->m_gExports.find(name);
      if( gep != _p->m_gExports.end() )
      {
         _p->m_gExports.erase( gep );
      }
      
      Private::GlobalsMap::iterator gp = _p->m_gSyms.find(name);
      if( gp != _p->m_gSyms.end() )
      {
         delete gp->second;
         _p->m_gSyms.erase( gp );
      }
      
      Private::DepMap::iterator dp = _p->m_depsBySymbol.find( name );
      if ( dp != _p->m_depsBySymbol.end() )
      {
         _p->m_depsBySymbol.erase( dp );
      }
   }
   
   // Remove all the related dependencies
   Private::DepList::iterator depi = _p->m_deplist.begin();
   while( depi != _p->m_deplist.end() ) 
   {
      Private::Dependency* dep = *depi;
      if( dep->m_idef == def )
      {
         depi = _p->m_deplist.erase( depi );
         delete dep;
         
      }
      else 
      {
         ++depi;
      }
   }
   
   // remove the definition
   Private::ImportDefList::iterator dli = 
            std::find( _p->m_importDefs.begin(), _p->m_importDefs.end(), def );
   if( dli != _p->m_importDefs.end() )
   {
      _p->m_importDefs.erase( dli );
   }
   
   // remove the nsImport, if any.
   Private::NSImportList::iterator nsi = _p->m_nsimports.begin();
   while( nsi != _p->m_nsimports.end() )
   {
      Private::NSImport* ns = *nsi;
      if( ns->m_def == def )
      {
         nsi = _p->m_nsimports.erase( nsi );
         delete ns;
      }
      else
      {
         ++nsi;
      }
   }
   
   delete def;
}


ImportDef* Module::addLoad( const String& name, bool bIsUri )
{
   ImportDef* id = new ImportDef;
   id->setLoad( name, bIsUri );
   
   ModRequest* req = 0;
   if( ! addModuleRequirement( id, req ) )
   {
      delete id;
      return 0;
   }
   
   _p->m_importDefs.push_back( id );   
   return id;
}

void Module::addImportRequest( Requirement* req, 
               const String& sourceMod, bool bModIsPath )
{
   ImportDef* id = new ImportDef;
   id->setDirect( req->name(), sourceMod, bModIsPath );
   if( sourceMod != "" )
   {
      ModRequest* mr;
      addModuleRequirement( id, mr );
   }
   _p->m_importDefs.push_back( id );   
   
   // add the dependency to the symbol.
   Private::Dependency* dep = new Private::Dependency( req->name() );
   dep->m_idef = id;
   dep->m_waitings.push_back( req );
   _p->m_deplist.push_back( dep );
}


void Module::addImportRequest( t_func_import_req cbFunc, const String& symName, 
         const String& sourceMod, bool bModIsPath )
{
   FuncRequirement* r = new FuncRequirement(symName, cbFunc);
   addImportRequest( r, sourceMod, bModIsPath );
}


Symbol* Module::addImplicitImport( const String& name, bool& isNew )
{
   static Class* symClass = Engine::instance()->symbolClass();
   static Collector* coll = Engine::instance()->collector();

   // We can't be called if the symbol is alredy declared elsewhere.
   Private::GlobalsMap::iterator pos = _p->m_gSyms.find( name );
   if( _p->m_gSyms.find( name ) != _p->m_gSyms.end() )
   {
      isNew = false;
      return pos->second;
   }

   isNew = true;
   // Record the fact that we have to save transform an unknown symbol...
   Symbol* uks = new Symbol( name, Symbol::e_st_extern, 0, 0);
   FALCON_GC_STORE( coll, symClass, uks );
   Private::Dependency* dep = new Private::Dependency(uks);
   dep->m_sourceName = name;
   _p->m_deplist.push_back(dep);
   _p->m_depsBySymbol[name] = dep;
   // ... and save the symbols.
   _p->m_gSyms[name] = uks;

   return uks;
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


Symbol* Module::addRequirement( Requirement* cr )
{
   const String& symName = cr->name();
   
   // we don't care if the symbol already exits; the mehtod would just return 0.
   Symbol* imported = addImplicitImport( symName );
   
   // is the symbol defined?
   if( imported->type() != Symbol::e_st_extern )
   {
      cr->onResolved( this, imported, this, imported );
      delete cr;
      return imported;
   }
    
   // at this point, the dependency must be created by implicit import.
   Private::Dependency* dep = _p->m_depsBySymbol[symName];
   if( dep == 0 )
   {
      // should not happen -- but if it happen, we can only search in global space
      dep = new Private::Dependency( imported );
      _p->m_depsBySymbol[imported->name()] = dep;
      _p->m_deplist.push_back(dep);      
   }
   
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waitings.push_back( cr );
   _p->m_reqslist.push_back( cr );
   
   return imported;
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



void Module::forwardNS( Module* mod, const String& remoteNS, const String& localNS )
{
   static Collector* coll = Engine::instance()->collector();
   static Class* symClass = Engine::instance()->symbolClass();
   
   Private::GlobalsMap& globs = mod->_p->m_gSyms;
   
   // search the symbols in the remote namespace.
   String nsPrefix = remoteNS + ".";
   Private::GlobalsMap::iterator gi = globs.lower_bound(nsPrefix);
   while( gi != globs.end() && gi->first.startsWith(nsPrefix) )
   {
      String localName = localNS + "." + gi->first.subString(nsPrefix.length());
      if( _p->m_gSyms.find(localName) == _p->m_gSyms.end())
      {
         Symbol* gs = new Symbol(*gi->second);
         FALCON_GC_STORE( coll, symClass, gs );
         _p->m_gSyms[localName] = gs;
      }
      ++gi;
   }
}


bool Module::addConstant( const String& name, const Item& value )
{
   Symbol* gsym = addVariable( name, true );
   if ( gsym == 0 ) 
   { 
      return false;
   }
   
   gsym->defaultValue( value );
   gsym->setConstant();
   return true;
}


void Module::checkWaitingFwdDef( Symbol* sym )
{   
   // ignore non-globals
   if ( sym->type() != Symbol::e_st_global )
   {
      return;
   }
   
   Private::DepMap::iterator pos = _p->m_depsBySymbol.find( sym->name() );
   if( pos != _p->m_depsBySymbol.end() )
   {
      // if the request covers an explicit import, this is an error.
      Private::Dependency* dep = pos->second;
      if( dep->m_idef != 0 )
      {
         throw new CodeError( 
            ErrorParam(e_already_def, sym->declaredAt(), m_uri ) 
            // TODO add source reference of the imported def
            .extra( String("imported symbol"))
            .origin(ErrorParam::e_orig_compiler)
            );
      }
      else
      {
         // a genuine forward definition
         Error* err = dep->onResolved( this, this, sym );

         // remove the dependency (was just a forward marker)
         _p->m_depsBySymbol.erase( pos );
         Private::DepList::iterator dli =
            std::find( _p->m_deplist.begin(), _p->m_deplist.end(), dep );
         if( dli != _p->m_deplist.end() )
         {
            _p->m_deplist.erase( dli );
         }
         delete dep;

         // throw in case of errors
         if( err != 0 )
         {
            throw err;
         }            
      }
   }
}


Function* Module::getMainFunction()
{
   if ( m_mainFunc == 0 )
   {
      // see if someone was nice enough to add it elsewhere.
      Private::FunctionMap::iterator iter = _p->m_functions.find("__main__");
      if ( iter != _p->m_functions.end() )
      {
         m_mainFunc = iter->second;
      }
      else {
         m_mainFunc = new SynFunc("__main__");
         _p->m_functions[m_mainFunc->name()] = m_mainFunc;
      }
   }
   
   return m_mainFunc;
}


void Module::setMainFunction( Function* mf )
{
   delete m_mainFunc;
   m_mainFunc = mf;
   mf->module(this);
   mf->name("__main__");
   
   _p->m_functions[mf->name()] = mf;
}

//=====================================================================
// Classes
//=====================================================================


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
         Item& value = pos->second->defaultValue();
         value.setUser( value.asClass(), hcls );
      }
   }
}

   
}

/* end of module.cpp */
