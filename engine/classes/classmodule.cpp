/*
   FALCON - The Falcon Programming Language.
   FILE: classmodule.cpp

   Module object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 22 Feb 2012 19:50:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classmodule.cpp"

#include <falcon/classes/classmodule.h>
#include <falcon/module.h>
#include <falcon/trace.h>
#include <falcon/module.h>
#include "../module_private.h"
#include "falcon/vmcontext.h"
#include "falcon/errors/ioerror.h"
#include <falcon/symbol.h>
#include <falcon/itemarray.h>
#include <falcon/modrequest.h>
#include <falcon/importdef.h>
#include <falcon/mantra.h>
#include <falcon/itemdict.h>
#include <falcon/stdhandlers.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/ioerror.h>
#include <falcon/errors/accesserror.h>

#include <vector>

namespace Falcon
{

ClassModule::ClassModule():
   Class("Module")
{
   m_getAttributeMethod.methodOf(this);
   m_setAttributeMethod.methodOf(this);
   m_clearPriority = 3;
}

ClassModule::~ClassModule()
{
}


void ClassModule::dispose( void* self ) const
{
   Module* mod = static_cast<Module*>(self);
   mod->decref();
}


void* ClassModule::clone( void* source ) const
{
   Module* mod = static_cast<Module*>(source);
   TRACE( "Cloning module %p (%s - %s)", mod, mod->name().c_ize(), mod->uri().c_ize() );
   
   Module* modcopy = new Module(*mod);
   return modcopy;
}


void* ClassModule::createInstance() const
{
   return new Module;
}


void ClassModule::enumerateProperties( void*, Class::PropertyEnumerator& cb ) const
{
   cb("attributes" );
   cb("name" );
   cb("uri");
}


void ClassModule::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{
   Module* mod = static_cast<Module*>(instance);

   Item i_name = mod->name();
   Item i_uri = mod->uri();

   cb("name", i_name );
   cb("uri", i_uri );
}


bool ClassModule::hasProperty( void*, const String& prop ) const
{
   return
         prop == "attributes"
         || prop == "getAttribute"
         || prop == "name"
         || prop == "setAttribute"
         || prop == "uri"
         ;
}

   
void ClassModule::store( VMContext*, DataWriter* stream, void* instance ) const
{
   Module* mod = static_cast<Module*>(instance);
   TRACE( "ClassModule::store -- Storing module %p %s (%s - %s)", 
      mod, (mod->isNative()?"native" : "syntactic" ),
      mod->name().c_ize(), mod->uri().c_ize() );
   
   stream->write( mod->isNative() );
   stream->write(mod->name());
   
   if( mod->isNative() )
   {
      stream->write( mod->uri() );
      return;
   }
   
   // otherwise, we don't have to save the URI, as it will be rewritten.
   // First, prepare to save the module ids.
   Module::Private* mp = mod->_p;
   int32 progID;
   
   // first, save symbols
   mod->globals().store(stream);
   
   // Now store the module requests
   {
      // first, number the module requests.
      Module::Private::ReqList& mrlist = mp->m_mrlist;
      progID = (int32) mrlist.size();
      TRACE1( "ClassModule::store -- storing %d mod requests", progID );
      stream->write( progID );
      Module::Private::ReqList::iterator mri = mrlist.begin();
      progID = 0;
      while( mri != mrlist.end() ) {
         ModRequest* req = *mri;
         req->store( stream );
         // save the progressive ID.
         req->id( progID++ );
         ++mri;
      }

      // We can now proceed to the import defs.
      Module::Private::ImportDefList& idlist = mp->m_importDefs;
      progID = (int32) idlist.size();
      TRACE1( "ClassModule::store -- storing %d import definitions", progID );
      stream->write( progID );
      Module::Private::ImportDefList::iterator idi = idlist.begin();
      progID = 0;
      while( idi != idlist.end() ) {
         ImportDef* def = *idi;
         def->id( progID++ );
         def->store( stream );
         if( def->modReq() != 0 )
         {
            stream->write( (int32) def->modReq()->id() );
         }
         else {
            stream->write( (int32) -1 );
         }

         // save the progressive ID.
         ++idi;
      }

      // finally, we can write the modRequest->import deps.
      mri = mrlist.begin();
      while( mri != mrlist.end() )
      {
         ModRequest* req = *mri;
         uint32 count = (uint32) req->importDefCount();
         TRACE1( "ClassModule::store -- Request %d has %d imports", req->id(), count );
         stream->write( count );
         for( uint32 i = 0; i < count; ++i ) {
            ImportDef* idl = req->importDefAt(i);
            uint32 id = idl->id();
            stream->write( id );
            TRACE2( "ClassModule::store -- Request %d -> import %d", req->id(), id );
         }
         ++mri;
      }
   }
   
   // namespace imports.
   {
      Module::Private::NSImportList& nsilist = mp->m_nsimports;
      progID = (int32) nsilist.size();
      stream->write( progID );
      TRACE1( "ClassModule::store -- storing %d namespace imports", progID );
      Module::Private::NSImportList::iterator nsi = nsilist.begin();
      progID = 0;
      while( nsi != nsilist.end() ) {
         Module::Private::NSImport* ipt = *nsi;
         stream->write( ipt->m_from );
         stream->write( ipt->m_to );
         if( ipt->m_def != 0 )
         {
            stream->write( (int32) ipt->m_def->id() );
         }
         else {
            stream->write( (int32) -1 );
         }

         ++nsi;
      }
   }

   // and finally, dependencies.
   {
      Module::Private::DepList& deplist = mp->m_deplist;
      progID = (int32) deplist.size();
      stream->write( progID );
      TRACE1( "ClassModule::store -- storing %d dependencies", progID );
      Module::Private::DepList::iterator depi = deplist.begin();
      progID = 0;
      while( depi != deplist.end() ) {
         Module::Private::Dependency* dep = *depi;
         dep->m_id = progID++;
         stream->write( dep->m_sourceName );
         stream->write( dep->m_targetName );
         if( dep->m_variable != 0 )
         {
            stream->write( (int32) dep->m_variable->id() );
         }
         else {
            stream->write( (int32)-1 );
         }

         if( dep->m_idef != 0 )
         {
            stream->write( (int32) dep->m_idef->id() );
         }
         else {
            stream->write( (int32)-1 );
         }            

         ++depi;
      }
   }
   MESSAGE1( "Module store attributes." );

   // restore the attributes
   mod->attributes().store(stream);
   
   MESSAGE1( "Module store complete." );
}


void ClassModule::restore( VMContext* ctx, DataReader* stream ) const
{
   static Class* mcls = Engine::handlers()->moduleClass();
   MESSAGE( "Restoring module..." );
   
   bool bIsNative;
   String name;
   stream->read( bIsNative );
   stream->read( name );
   
   TRACE1( "Module being restored: %s (%s)", 
      (bIsNative?"native" : "syntactic" ),
      name.c_ize() );
   
   if( bIsNative )
   {
      String origUri;
      stream->read( origUri );
      
      Module* mod = new Module( name, true );
      mod->uri( origUri );
      ctx->pushData( FALCON_GC_STORE( mcls, mod ) );
      return;
   }
   
   // 
   Module* mod = new Module(name, false);
   
   try {
      restoreModule( mod, stream );
   }
   catch( ... )
   {
      delete mod;
      throw;
   }
 
   ctx->pushData( Item( mcls, mod ) );
}


void ClassModule::restoreModule( Module* mod, DataReader* stream ) const
{
   TRACE( "ClassModule::restoreModule %s", mod->name().c_ize() );
    
   int32 progID, count;
   // First, prepare to save the module ids.
   Module::Private* mp = mod->_p;
      
   // first, write the symbols.
   mod->globals().restore(stream);
   
   // Now restore the module requests
   Module::Private::ReqList& mrlist = mp->m_mrlist;
   Module::Private::ReqMap& mrmap = mp->m_mrmap;

   stream->read( count );
   TRACE1( "ClassModule::restoreModule -- reading %d mod requests", count );
   //mrlist.reserve( count );
   progID = 0;
   while( progID < count ) {
      ModRequest* req = new ModRequest;
      try
      {
         req->restore( stream );
         req->id(progID);
         TRACE2( "Read mod request %s (%s) as %s",
            req->name().c_ize(),
            (req->isUri() ? " by uri" : "by name" ),
            (req->isLoad() ? "load" : "import" ) );
      }  
      catch( ... )
      {
         delete req;
         throw;
      }

      mrlist.push_back( req );
      mrmap[req->name()] = req;
      ++progID;
   }

   // We can now proceed to the import defs.
   Module::Private::ImportDefList& idlist = mp->m_importDefs;

   stream->read( count );
   TRACE1( "ClassModule::restoreModule -- reading %d import defs", count );
   //idlist.reserve( count );
   progID = 0;
   while( progID < count ) 
   {
      ImportDef* def = new ImportDef;
      try {
         def->restore( stream );
         int32 modreq = -1;
         stream->read( modreq );
         if( modreq >= 0 )
         {
            if( modreq >= (int32) mrlist.size() )
            {
               throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
                  .origin( ErrorParam::e_orig_loader )
                  .extra(String("Module request ID out of range on ImportDef ").N(progID) )
                  );
            }
            
            def->modReq( mrlist[modreq] );
            def->id(progID);
         }
         
         idlist.push_back(def);
      }
      catch( ... ) {
         delete def;
         throw;
      }
      
      ++progID;
   }
   
   // finally, we can load the modRequest->import deps.
   {
      Module::Private::ReqList::iterator mri = mrlist.begin();
      while( mri != mrlist.end() )
      {
         ModRequest* req = *mri;
         uint32 count;
         stream->read( count );
         TRACE1( "ClassModule::restoreModule -- Request %d has %d imports", req->id(), count );
         bool general = false;
         for( uint32 i = 0; i < count; ++i ) {
            uint32 id;
            stream->read(id);
            if ( id >= idlist.size() ) {
               throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
                        .origin( ErrorParam::e_orig_loader )
                        .extra(String("ImportDef ID out of range on ModReq ").N(req->id()) )
                        );
            }
            ImportDef* idef = idlist[id];
            req->addImportDef(idef);

            // eventually, save the module as a generic provider.
            if( idef->isGeneric() && ! general )
            {
               TRACE1( "ClassModule::restoreModule -- Module %s is declared generic", req->name().c_ize() );
               general = true;
               mp->m_genericMods.push_back( req );
            }
         }
         ++mri;
      }
   }

   // namespace imports.
   Module::Private::NSImportList& nsilist = mp->m_nsimports;
   stream->read( count );
   TRACE1( "ClassModule::restoreModule -- reading %d namespaces", count );
   progID = 0;
   while( progID < count ) 
   {
      String sFrom, sTo;
      int32 defID;
      
      stream->read( sFrom );
      stream->read( sTo );
      stream->read( defID );
      ImportDef* idef = 0;
      
      if( defID >= 0 )
      {
         if( defID >= (int32) idlist.size() )
         {
            throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
               .origin( ErrorParam::e_orig_loader )
               .extra(String("Import ID out of range on Namespace import ").N(progID) )
               );
         }
         idef = idlist[defID];
      }
      
      Module::Private::NSImport* nsi = new Module::Private::NSImport(idef, sFrom, sTo );
      nsi->m_bPerformed = false;
      nsilist.push_back( nsi );

      ++progID;
   }

   // and finally, dependencies.
   Module::Private::DepList& deplist = mp->m_deplist;
   stream->read( count );
   TRACE1( "ClassModule::restoreModule -- reading %d dependencies", count );   
   progID = 0;
   while( progID < count ) 
   {
      String sSourceName, sName;
      int32 idSymbol, idDef;
      
      stream->read( sSourceName );
      stream->read( sName );
      stream->read( idSymbol );
      stream->read( idDef );
      

      ImportDef* idef = 0;
      
      if( idDef >= 0 )
      {
         if( idDef >= (int32) idlist.size() )
         {
            throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
               .origin( ErrorParam::e_orig_loader )
               .extra(String("ImportDef out of range dependency ").N(progID) )
               );
         }
         idef = idlist[idDef];
      }
      
      if( idSymbol >= 0 )
      {
         if( idSymbol >= (int32) mod->globals().size() )
         {
            throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
               .origin( ErrorParam::e_orig_loader )
               .extra(String("Symbol out of range dependency ").N(progID) )
               );
         }         
      }
      
      Module::Private::Dependency* dep = new Module::Private::Dependency( sName );
      dep->m_idef = idef;
      dep->m_sourceName = sSourceName;
      dep->m_targetName = sName;
      VarDataMap::VarData* vd = 0;
      if( idSymbol >= 0 ) {
         vd = mod->globals().getGlobal(idSymbol);
         dep->m_variable = &vd->m_var;
      }
      else {
         dep->m_variable = 0;
      }
      dep->m_id = progID;

      deplist.push_back( dep );      
      mp->m_depsByName[sName] = dep;

      TRACE2( "ClassModule::restoreModule -- restored dependency %d: %s (var:%s, idef:%d)",
               progID, sName.c_ize(), (vd == 0? "<undef>": vd->m_name.c_ize()), idDef );

      ++progID;
   }
   
   MESSAGE1( "Module restore -- attributes" );

   // restore the attributes
   mod->attributes().restore(stream);

   MESSAGE1( "Module restore complete." );
}

void ClassModule::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   Module* mod = static_cast<Module*>(instance);
   TRACE( "Flattening module %p %s (%s - %s)", 
      mod,  (mod->isNative()?"native" : "syntactic" ),
      mod->name().c_ize(), mod->uri().c_ize() );
   
   if( mod->isNative() )
   {
      // nothing to do,
      return;
   }
      
   // First, get enough lenght
   Module::Private* mp = mod->_p;
   
   subItems.reserve( 
      mod->globals().size() +
      mp->m_mantras.size()+
      mp->m_reqslist.size() +
      mod->attributes().size() * 2 +
      4
   );
   
   // save all the global variables
   mod->globals().flatten(ctx, subItems);
   TRACE( "ClassModule::flatten -- stored %d variables", (uint32) subItems.length() );
   
   // save mantras
   {
      Module::Private::MantraMap& mantras = mp->m_mantras;
      Module::Private::MantraMap::iterator fi = mantras.begin();
      while( fi != mantras.end() ) 
      {
         TRACE1("Flattening mantra %s", fi->first.c_ize() );
         Mantra* mantra = fi->second;
         // skip hyperclasses
         if( ! mantra->isCompatibleWith( Mantra::e_c_hyperclass ))
         {
            Class* cls = mantra->handler();
            TRACE1("Mantra %s has handler %s(%p)", fi->first.c_ize(), cls->name().c_ize(), cls );
            subItems.append( Item(cls, mantra) );
         }
         ++fi;
      }
   }   
   // Push a nil as a separator
   subItems.append( Item() );
   
   {
      Module::Private::RequirementList& reqs = mp->m_reqslist;
      Module::Private::RequirementList::iterator reqi = reqs.begin();
      while( reqi != reqs.end() ) {
         Requirement* req = *reqi;
         subItems.append( Item(req->cls(), req) );
         ++reqi;
      }
   }

   // Push a nil as a separator
   subItems.append( Item() );
   
   // finally push the classes in need of init
   {
      Module::Private::InitList& initList = mp->m_initList;
      Module::Private::InitList::iterator ii = initList.begin();
      while( ii != initList.end() ) {
         Class* cls = *ii;
         subItems.append( Item(cls->handler(), cls) );
         ++ii;
      }
   }
   // Push a nil as a separator
   subItems.append( Item() );

   // save the attributes.
   mod->attributes().flatten(subItems);

   // Push a nil as a separator
   subItems.append( Item() );

   // complete.
}


void ClassModule::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   Module* mod = static_cast<Module*>(instance);
   TRACE( "ClassModule::unflatten -- module %p %s (%s - %s)",
      mod,  (mod->isNative()?"native" : "syntactic" ),
      mod->name().c_ize(), mod->uri().c_ize() );
   
   if( mod->isNative() )
   {
      // nothing to do,
      return;
   }
      
   Module::Private* mp = mod->_p;
   uint32 pos = 0;
      
   // First, restore the global variables.
   mod->globals().unflatten( ctx, subItems, 0, pos);
   TRACE( "ClassModule::unflatten -- restored %d globals", pos );

   const Item* current = &subItems[pos];
   while( ! current->isNil() && pos < subItems.length()-2 )
   {
      Mantra* mantra = static_cast<Mantra*>(current->asInst());  
      TRACE1( "ClassModule::unflatten -- storing mantra %s ", mantra->name().c_ize() );

      if( mantra->name() == "__main__" )
      {
         mod->m_mainFunc = static_cast<Function*>(mantra);
      }

      mp->m_mantras[mantra->name()] = mantra;
      mantra->module( mod );
      // no need to store the mantra in globals:
      // the globals already unflattened and mantras are in place.
      
      ++pos;
      current = &subItems[pos];
   }
   
   TRACE( "ClassModule::unflatten -- restored mantras, at position %d", pos );
   
   Module::Private::RequirementList& reqs = mp->m_reqslist;
   current = &subItems[++pos];
   while( ! current->isNil() && pos < subItems.length()-1 )
   {      
      Requirement* req = static_cast<Requirement*>(current->asInst());
      TRACE1( "ClassModule::unflatten -- restored requirement for %s", req->name().c_ize() );
      reqs.push_back( req );
      // find the dependency to which this requirement belongs to
      Module::Private::DepMap::iterator depi = mp->m_depsByName.find( req->name() );
      if( depi != mp->m_depsByName.end() ) {
         TRACE2( "ClassModule::unflatten -- found dependency for requirement %s", req->name().c_ize() );
         depi->second->m_waitings.push_back( req );
      }
      ++pos;
      current = &subItems[pos];
   }
   
   // recover init classes
   Module::Private::InitList& inits = mp->m_initList;
   current = &subItems[++pos];
   while( ! current->isNil() && pos < subItems.length()-1)
   {
      Class* cls = static_cast<Class*>(current->asInst());
      TRACE1( "ClassModule::unflatten -- restored class in need of init %s", cls->name().c_ize() );
      inits.push_back( cls );
      ++pos;
      current = &subItems[pos];
   }

   TRACE( "ClassModule::unflatten -- restored init classes, at position %d", pos );

   mod->attributes().unflatten( subItems, pos );

   TRACE( "ClassModule::unflatten -- restored attributes, at position %d", pos );
}

   
void ClassModule::describe( void* instance, String& target, int , int ) const
{
   Module* mod = static_cast<Module*>(instance);
   target = "Module " + mod->name();
}

void ClassModule::gcMarkInstance( void* instance, uint32 mark ) const
{
   static_cast<Module*>(instance)->gcMark(mark);
}

bool ClassModule::gcCheckInstance( void* instance, uint32 mark ) const
{
   return static_cast<Module*>(instance)->lastGCMark() >= mark;
}

bool ClassModule::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   // NAME - URI
   Item& i_name = ctx->opcodeParam(0);
   Item& i_uri = ctx->opcodeParam(1);
   
   if( pcount < 1 
      || ! i_name.isString() 
      || (pcount > 1 && ! i_uri.isString()) 
      )
   {
      throw new ParamError( ErrorParam(e_inv_params, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)
         .extra( "S,[S]") );
   }
   
   Module* module =  static_cast<Module*>(instance);
   module->name( *i_name.asString() );
   if( pcount > 1 )
   {
      module->uri( *i_uri.asString() );
   }
   
   return false;
}


void ClassModule::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   Module* mod = static_cast<Module*>(instance);

   if( prop == "attributes" )
   {
      ItemDict* dict = new ItemDict;
      uint32 size = mod->attributes().size();
      for( uint32 i = 0; i < size; ++i ) {
         Attribute* attr = mod->attributes().get(i);
         dict->insert( FALCON_GC_HANDLE( new String(attr->name())), attr->value() );
      }
      ctx->stackResult(1, FALCON_GC_HANDLE(dict) );
   }
   else if( prop == "getAttribute" )
   {
      ctx->topData().methodize(&m_getAttributeMethod);
   }
   else if( prop == "name" )
   {
      ctx->stackResult(1, FALCON_GC_HANDLE( new String(mod->name())) );
   }
   else if( prop == "setAttribute" )
   {
      ctx->topData().methodize(&m_setAttributeMethod);
   }
   else if( prop == "uri" )
   {
      ctx->stackResult(1, FALCON_GC_HANDLE( new String(mod->uri())) );
   }
   else {
      Class::op_getProperty(ctx, instance, prop );
   }
}

//========================================================================
// Methods
//

ClassModule::GetAttributeMethod::GetAttributeMethod():
   Function("getAttribute")
{
   signature("S");
   addParam("name");
}

ClassModule::GetAttributeMethod::~GetAttributeMethod()
{}


void ClassModule::GetAttributeMethod::invoke( VMContext* ctx, int32 )
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_name = ctx->param(0);
   if( ! i_name->isString() )
   {
      ctx->raiseError(paramError());
      return;
   }

   const String& attName = *i_name->asString();
   Module* mantra = static_cast<Module*>(self.asInst());
   Attribute* attr = mantra->attributes().find(attName);
   if( attr == 0 )
   {
      ctx->raiseError( new AccessError( ErrorParam(e_dict_acc, __LINE__, SRC )
            .symbol("Module.getAttribute")
            .module("[core]")
            .extra(attName) ) );
      return;
   }

   ctx->returnFrame(attr->value());
}

ClassModule::SetAttributeMethod::SetAttributeMethod():
   Function("setAttribute")
{
   signature("S,[X]");
   addParam("name");
   addParam("value");
}

ClassModule::SetAttributeMethod::~SetAttributeMethod()
{}


void ClassModule::SetAttributeMethod::invoke( VMContext* ctx, int32 )
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_name = ctx->param(0);
   Item* i_value = ctx->param(1);
   if( ! i_name->isString() )
   {
      ctx->raiseError(paramError());
      return;
   }

   const String& attName = *i_name->asString();
   Module* mantra = static_cast<Module*>(self.asInst());

   if( i_value == 0 )
   {
      mantra->attributes().remove(attName);
   }
   else {
      Attribute* attr = mantra->attributes().find(attName);
      if( attr == 0 )
      {
         attr = mantra->attributes().add(attName);
      }

      attr->value().assign( *i_value );
   }

   ctx->returnFrame();
}


}

/* end of classmodule.cpp */
