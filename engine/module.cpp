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

   virtual void onResolved( const Module* sourceModule, const String& sourceName, Module* targetModule, const Item& value, const Variable* targetVar )
   {
      Error* err = m_cbFunc( sourceModule, sourceName, targetModule, value, targetVar );
      if( err != 0 ) throw err;
   }
   
   // This applies only to native modules, which doesn't store requirementes.
   virtual Class* cls() const { return 0; }
   
private:
   t_func_import_req m_cbFunc;   
};
   
   
Error* Module::Private::Dependency::onResolved( Module* sourceMod, Module* hostMod, Item* source )
{
   Error* res = 0;

   bool firstError = true;
   Private::Dependency::WaitingList::iterator iter = m_waitings.begin();
   while( m_waitings.end() != iter )
   {
      try 
      {
         Requirement* req = *iter;
         req->onResolved( sourceMod, m_sourceName, hostMod, *source, m_variable );
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
               res = new LinkError( ErrorParam( e_link_error, 0, hostMod->uri())
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
      //delete *id_i;
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

   MantraMap::iterator mi = m_mantras.begin();
   while( mi != m_mantras.end() )
   {
      Mantra* mantra = mi->second;
      delete mantra;
      ++mi;
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
   m_anonMantras(0),
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
   m_anonMantras(0),
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
   m_anonMantras(0),
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
   unload();
   // this is doing to do a bit of stuff; see ~Private()
   delete _p;
   TRACE("Module '%s' deletion complete", m_name.c_ize() );
}


uint32 Module::depsCount() const
{
   return _p->m_importDefs.size();
}

ImportDef* Module::getDep( uint32 n ) const
{
   return _p->m_importDefs[n];
}


void Module::gcMark( uint32 mark )
{
   if( m_lastGCMark != mark )
   {
      TRACE( "Module::gcMark -- marking %s", name().c_ize() );

      m_lastGCMark = mark;
      if ( m_modSpace != 0 )
      {
         m_modSpace->gcMark( mark );
      }
      
      m_globals.gcMark(mark);
      /*
      Private::MantraMap::iterator mi = _p->m_mantras.begin();
      Private::MantraMap::iterator mi_end = _p->m_mantras.end();

      while( mi != mi_end ) {
         Mantra* m = mi->second;
         m->gcMark(mark);
         ++mi;
      }
      */
   }
}


bool Module::promoteExtern( Variable* ext, const Item& value, int32 redeclaredAt )
{
   if( ! m_globals.promoteExtern(ext->id(), value, redeclaredAt) ) {
      return false;
   }

   // see if this covers a forward declaration.
   VarDataMap::VarData* vd = m_globals.getGlobal(ext->id());
   fassert( vd != 0 ); // promotion to extern would have failed.
   if( ! checkWaitingFwdDef( vd->m_name, vd->m_data ) ) {
      return false;
   }

   return true;
}


bool Module::resolveExternValue( const String& name, Module* source, Item* value )
{
   // was a dependency waiting for this?
   Private::DepMap::const_iterator diter = _p->m_depsByName.find( name );
   if( diter != _p->m_depsByName.end() ) {
      Error* err = diter->second->onResolved( source, this, value );
      if( err != 0 ) throw err;
   }

   // check if we have a variable associated with this.
   VarDataMap::VarData* vd = m_globals.getGlobal(name);
   if( vd == 0 ) {
      return false;
   }

   if( ! m_globals.promoteExtern(vd->m_var.id(), *value, 0) ) {
      return false;
   }

   return true;
}



Variable* Module::importValue( const String& name, Module* source, Item* value )
{
   bool bAlready = false;
   Variable* var = addImplicitImport(name, bAlready);
   if( var->type() == Variable::e_nt_extern )
   {
      // ok, we have an extern variable.
      // If it was already defined, there might be a dependency waiting for that.
      if( bAlready )
      {
         // was a dependency waiting for this?
         Private::DepMap::const_iterator diter = _p->m_depsByName.find( name );
         if( diter != _p->m_depsByName.end() ) {
            Error* err = diter->second->onResolved( source, this, value );
            if( err != 0 ) throw err;
         }
      }

      return var;
   }

   return 0;
}


void Module::addAnonMantra( Mantra* f )
{
   String name;
   do
   {
      name = "_anon#";
      name.N(m_anonMantras++);
   } while( _p->m_mantras.find( name ) != _p->m_mantras.end() );

   f->name( name );
   _p->m_mantras[f->name()] = f;
   f->module(this);

}


Variable* Module::addMantra( Mantra* f, bool bExport)
{
   TRACE(" Module::addMantra -- (%s(%p), %s, %d)",
            f->name().size() == 0 ? "(anon)" : f->name().c_ize(),
            f, bExport? "export" : "private", f->declaredAt() );

   //static Engine* eng = Engine::instance();

   VarDataMap::VarData* vd = m_globals.getGlobal( f->name() );
   if( vd != 0 && vd->m_var.type() != Variable::e_nt_extern  )
   {
      // already defined.
      TRACE1(" Module::addMantra -- %s(%p) already defined", f->name().c_ize(), f );
      return 0;
   }
   
   // add to the function vector so that we can account it.
   _p->m_mantras[f->name()] = f;
   f->module(this);
   Item value( f->handler(), f );

   // then add the required global.
   if( vd == 0 )
   {
      TRACE1(" Module::addMantra -- %s(%p) adding as new global", f->name().c_ize(), f );
      vd = m_globals.addGlobal( f->name(), value, bExport );
      fassert( vd != 0 );
   }
   else {
      TRACE1(" Module::addMantra -- %s(%p) promoting from extern.", f->name().c_ize(), f );
      if( ! promoteExtern( &vd->m_var, value, f->declaredAt() ) ) {
         TRACE1(" Module::addMantra -- %s(%p) promotion failed.", f->name().c_ize(), f );
         return false;
      }
   }
   
   vd->m_var.declaredAt( f->declaredAt() );
   return &vd->m_var;
}


Variable* Module::addFunction( const String &name, ext_func_t f, bool bExport )
{
   // check if the name is free.
   VarDataMap::VarData* vd = m_globals.getGlobal( name );
   if( vd == 0 )
   {
      return 0;
   }

   // ok, the name is free; add it
   Function* extfunc = new ExtFunc( name, f, this );
   return addMantra( extfunc, bExport );
}


Variable* Module::addSingleton( Class*, bool )
{
   // TODO
   return 0;
}


Variable* Module::addConstant( const String& name, const Item& value, bool bExport )
{
   VarDataMap::VarData* vd = m_globals.addGlobal( name, value, bExport );
   if( vd == 0 ) {
      return 0;
   }

   vd->m_var.setConst(true);
   return &vd->m_var;
}


Mantra* Module::getMantra( const String& name, Mantra::t_category cat ) const
{
   Private::MantraMap::const_iterator iter = _p->m_mantras.find(name);
   if( iter != _p->m_mantras.end() )
   {
      Mantra* mantra = iter->second;
      if( mantra->isCompatibleWith( cat ) )
      {
         return mantra;
      }
   }
   
   return 0;
}


/** Enumerate all functions and classes in this module.
*/
void Module::enumerateMantras( Module::MantraEnumerator& rator ) const
{
   Private::MantraMap::const_iterator iter = _p->m_mantras.begin();
   Private::MantraMap::const_iterator end = _p->m_mantras.end();

   while( iter != end )
   {
      const Mantra* m = iter->second;
      if( ! rator( *m, ++iter == end) )
         break;
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
         if ( m_globals.getGlobal( name ) != 0 )
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
          addImplicitImport( name );
          _p->m_depsByName[name]->m_idef = def;
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
      m_globals.removeGlobal( name );

      Private::DepMap::iterator dp = _p->m_depsByName.find( name );
      if ( dp != _p->m_depsByName.end() )
      {
         _p->m_depsByName.erase( dp );
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


Error* Module::addLoad( const String& name, bool bIsUri )
{
   ImportDef* id = new ImportDef;
   id->setLoad( name, bIsUri );
   
   ModRequest* req = 0;
   Error* err = addModuleRequirement( id, req );
   if (err != 0)
   {
      delete id;
      return err;
   }
   
   _p->m_importDefs.push_back( id );   
   return 0;
}

void Module::addImportRequest( Requirement* req, 
               const String& sourceMod, bool bModIsPath )
{
   ImportDef* id = 0;

   if( sourceMod != "" )
   {
      ImportDef* id = new ImportDef;
      id->setDirect( req->name(), sourceMod, bModIsPath );
      ModRequest* mr;
      addModuleRequirement( id, mr );
      _p->m_importDefs.push_back( id );
   }
   
   // add the dependency to the symbol.
   Private::Dependency* dep = new Private::Dependency( req->name() );
   dep->m_idef = id;
   dep->m_waitings.push_back( req );
   _p->m_deplist.push_back( dep );
   _p->m_depsByName[req->name()] = dep;
}


void Module::addImportRequest( t_func_import_req cbFunc, const String& symName, 
         const String& sourceMod, bool bModIsPath )
{
   FuncRequirement* r = new FuncRequirement(symName, cbFunc);
   addImportRequest( r, sourceMod, bModIsPath );
}


Variable* Module::addImplicitImport( const String& name, bool& isNew )
{
   // We can't be called if the symbol is already declared elsewhere.
   VarDataMap::VarData* vd = m_globals.getGlobal(name);
   if( vd != 0 )
   {
      isNew = false;
      return &vd->m_var;
   }

   isNew = true;

   // store a space for an external value in the global values vector.
   vd = m_globals.addExtern( name, 0 );

   Private::Dependency* dep = new Private::Dependency(name, &vd->m_var);
   _p->m_deplist.push_back(dep);
   _p->m_depsByName[name] = dep;

   return &vd->m_var;
}


Variable* Module::addRequirement( Requirement* cr )
{
   const String& symName = cr->name();
   
   // we don't care if the symbol already exits; the method would just return 0.
   Variable* imported = addImplicitImport( symName );
   
   // is the symbol defined?
   if( imported->type() != Variable::e_nt_extern )
   {
      try {
         Item* value = m_globals.getGlobalValue(imported->id());
         fassert( value != 0 );
         cr->onResolved( this, symName, this, *value, imported );
         delete cr;
         return imported;
      }
      catch (...) {
         delete cr;
         throw;
      }
   }
    
   // at this point, the dependency must be created by implicit import.
   Private::Dependency* dep = _p->m_depsByName[symName];
   if( dep == 0 )
   {
      // should not happen -- but if it happen, we can only search in global space
      dep = new Private::Dependency( symName, imported );
      _p->m_depsByName[symName] = dep;
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
      ul->unload();
   }
}


void Module::forwardNS( Module* mod, const String& remoteNS, const String& localNS )
{
   m_globals.forwardNS( &mod->m_globals, remoteNS, localNS );
}


bool Module::checkWaitingFwdDef( const String& name, Item* value )
{
   Private::DepMap::iterator pos = _p->m_depsByName.find( name );
   if( pos != _p->m_depsByName.end() )
   {
      // if the request covers an explicit import, this is an error.
      Module::Private::Dependency* dep = pos->second;
      if( dep->m_idef != 0 )
      {
         return false;
      }
      else
      {
         // a genuine forward definition;
         // by setting the symbol as not extern anymore, we prevent an useless promotion
         Error* err = dep->onResolved( this, this, value );

         // remove the dependency (was just a forward marker)
         _p->m_depsByName.erase( pos );
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

   return true;
}


Function* Module::getMainFunction()
{
   return m_mainFunc;
}


void Module::setMainFunction( Function* mf )
{
   m_mainFunc = mf;
   mf->module(this);
   mf->name("__main__");
   addMantra( mf, false );
}

//=====================================================================
// Classes
//=====================================================================

void Module::completeClass(FalconClass* fcls)
{                  
   // Completely resolved!
   if( !fcls->construct() )
   {
      // was not a full falcon class; we must change it into an hyper class.
      HyperClass* hcls = fcls->hyperConstruct();
      fassert2( hcls != 0, "called completeClass on an incomplete class");
      
      _p->m_mantras[hcls->name()] = hcls;
      // save the old falcon class under another name; we need to reference it.
      _p->m_mantras["$" + fcls->name()] = fcls;
      
      // anonymous classes cannot have a name in the global symbol table, so...
      Item* clsItem = m_globals.getGlobalValue( hcls->name() );
      if( clsItem != 0 )
      {
         clsItem->setUser( hcls->handler(), hcls );
      }
   }
}

   
}

/* end of module.cpp */
