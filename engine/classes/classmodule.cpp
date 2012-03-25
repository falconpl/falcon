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

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/ioerror.h>

#include <vector>

namespace Falcon
{

ClassModule::ClassModule():
   Class("Module")
{
}

ClassModule::~ClassModule()
{
}


void ClassModule::dispose( void* self ) const
{
   Module* mod = static_cast<Module*>(self);
   mod->unload();
   delete mod;
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
   
   int32 progID;
   // First, prepare to save the module ids.
   Module::Private* mp = mod->_p;
   
   // first, count the symbols
   {
      Module::Private::GlobalsMap& globs = mp->m_gSyms;
      progID = (int32) globs.size();
      TRACE1( "ClassModule::store -- storing %d symbols", progID );
      stream->write( progID );
      Module::Private::GlobalsMap::iterator globi = globs.begin();
      progID = 0;
      while( globi != globs.end() ) {
         Symbol* sym = globi->second;
         // reset the ID here
         sym->localId( progID ); 

         ++globi;
         ++ progID;
      }

      // also the exported
      Module::Private::GlobalsMap& exps = mp->m_gExports;
      progID = (int32) exps.size();
      TRACE1( "ClassModule::store -- storing %d export symbols", progID );
      stream->write( progID );
      Module::Private::GlobalsMap::iterator expi = exps.begin();
      while( expi != exps.end() ) {
         Symbol* sym = expi->second;
         // just write the references
         stream->write( (int32) sym->localId() );      
         ++expi;
      }
   }
   
   // Now store the module requests
   {
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
         if( dep->m_symbol != 0 )
         {
            stream->write( (int32) dep->m_symbol->localId() );
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
   
   MESSAGE1( "Module store complete." );
}


void ClassModule::restore( VMContext*, DataReader* stream, void*& empty ) const
{
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
      empty = mod;
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
 
   empty = mod;
}


void ClassModule::restoreModule( Module* mod, DataReader* stream ) const
{
   TRACE( "ClassModule::restoreModule %s", mod->name().c_ize() );
    
   int32 progID, count;
   // First, prepare to save the module ids.
   Module::Private* mp = mod->_p;
      
   // first, write the symbols.
   stream->read(count);      
   TRACE1( "ClassModule::restoreModule -- reading %d symbols", count );
   mp->m_symCount = count;
   progID = 0;
   

   // also the exported
   stream->read( count );
   TRACE1( "ClassModule::restoreModule -- reading %d exported symbols", count );
   progID = 0;
   while( progID < count ) {
      int32 symPos;
      // just write the references
      stream->read( symPos );   
      TRACE2( "Symbol %d is exported", symPos );
      if( symPos >= (int32) mp->m_symCount )
      {
         throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_loader )
            .extra(String("Exported symbol ID out of range on exported symbol ").N(progID) )
            );
      }

      mp->m_tempExport.push_back(symPos);
      ++progID;
   }
   
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
      
      // save the progressive ID.
      ++progID;
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
      String sName;
      int32 idSymbol, idDef;
      
      stream->read( sName );
      stream->read( idSymbol );
      stream->read( idDef );
      
      Symbol* sym = 0;
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
         if( idSymbol >= (int32) mp->m_symCount )
         {
            throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
               .origin( ErrorParam::e_orig_loader )
               .extra(String("Symbol out of range dependency ").N(progID) )
               );
         }         
      }
      sym = (Symbol*) idSymbol;
      
      Module::Private::Dependency* dep = new Module::Private::Dependency( sName );
      dep->m_idef = idef;
      dep->m_symbol = sym;      
      deplist.push_back( dep );      
      
      ++progID;
   }
   
   MESSAGE1( "Module restore complete." );
}

void ClassModule::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   static Class* clsSymbol = Engine::instance()->symbolClass();
   
   Module* mod = static_cast<Module*>(instance);
   TRACE( "Flattening module %p %s (%s - %s)", 
      mod,  (mod->isNative()?"native" : "syntactic" ),
      mod->name().c_ize(), mod->uri().c_ize() );
   
   if( mod->isNative() )
   {
      // nothing to do,
      return;
   }
      
   // First, prepare to save the module ids.
   Module::Private* mp = mod->_p;
   
   subItems.reserve( 
      mp->m_gSyms.size()*2 +
      mp->m_mantras.size()+
      mp->m_reqslist.size() +
      3
   );
   
   // First, save the symbols and their values.   
   {
      Module::Private::GlobalsMap& globs = mp->m_gSyms;
      Module::Private::GlobalsMap::iterator globi = globs.begin();
      while( globi != globs.end() ) {
         Symbol* sym = globi->second;
         TRACE1( "Flattening symbol %s in module %s", sym->name().c_ize(), mod->name().c_ize() );
         subItems.append(Item( clsSymbol, sym ) );
         
         const Item* value = sym->getValue( ctx );
         if( value != 0 )
         {
            if( ! value->isUser() || !static_cast<Class*>(value->asInst())->isCompatibleWith(Mantra::e_c_hyperclass))
            {
               // skip hyperclasses
               subItems.append( *value );
            }
            else { 
               subItems.append( Item() );
            }
         }
         else {
            subItems.append( Item() );
         }
         
         ++globi;
      }
   }
   
   {
      Module::Private::MantraMap& mantras = mp->m_mantras;
      Module::Private::MantraMap::iterator fi = mantras.begin();
      while( fi != mantras.end() ) 
      {
         Mantra* mantra = fi->second;
         // skip hyperclasses
         if( ! mantra->isCompatibleWith( Mantra::e_c_hyperclass ))
         {
            subItems.append( Item(mantra->handler(), mantra) );
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
   
   // complete.
}


void ClassModule::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   Module* mod = static_cast<Module*>(instance);
   TRACE( "Unflattening module %p %s (%s - %s)", 
      mod,  (mod->isNative()?"native" : "syntactic" ),
      mod->name().c_ize(), mod->uri().c_ize() );
   
   if( mod->isNative() )
   {
      // nothing to do,
      return;
   }
      
   Module::Private* mp = mod->_p;
   uint32 pos = 0;
      
   // First, restore the symbol values.
   //exps[ exported->name() ] = (Symbol*) symPos;
   std::vector<Symbol*> syms;
   syms.reserve( mp->m_symCount );
   Module::Private::GlobalsMap& globs = mp->m_gSyms;
   while( pos < mp->m_symCount*2 ) 
   {         
      const Item& valueSym = subItems[pos++];
      fassert( valueSym.isUser() );      
      Symbol* sym = static_cast<Symbol*>(valueSym.asInst());
      syms.push_back( sym );
      globs[sym->name()] = sym;
      
      Item& value = subItems[pos++];
      // don't bother with nil values, as they could be associated with externs
      if( ! value.isNil() )
      {
         sym->setValue( ctx, value );
      }
   }
   
   // unflatten exports and dependencies.
   {      
      Module::Private::GlobalsMap& exps = mp->m_gExports;
      std::vector<int32>& expids = mp->m_tempExport;
      std::vector<int32>::iterator expi = expids.begin();
      while( expi != expids.end() ) {
         int32 pos = *expi;
         fassert( (uint32)pos < syms.size() ); // already checked in restore
         Symbol* sym = syms[pos];
         exps[sym->name()] = sym;
         
         ++expi;
      }
      
      Module::Private::DepList& deplist = mp->m_deplist;
      Module::Private::DepList::iterator depi = deplist.begin();
      while( depi != deplist.end() ) {
         Module::Private::Dependency* dep = *depi;
         int32 pos = (int32) dep->m_symbol;
         if( pos >= 0 )
         {
            fassert( (uint32)pos < syms.size() ); // already checked in restore
            Symbol* sym = syms[pos];
            dep->m_symbol = sym;
            mp->m_depsBySymbol[ sym->name() ] = dep;
         }
         else {
            dep->m_symbol = 0;
         }
         
         ++depi;
      }
   }
   
   const Item* current = &subItems[pos];
   while( ! current->isNil() && pos < subItems.length()-2 )
   {
      Mantra* mantra = static_cast<Mantra*>(current->asInst());  
      if( mantra->name() == "__main__" )
      {
         mod->setMainFunction( static_cast<Function*>(mantra) );
      }
      else {
         mp->m_mantras[mantra->name()] = mantra;
      }
      mantra->module( mod );
      
      ++pos;
      current = &subItems[pos];
   }
   
   
   Module::Private::RequirementList& reqs = mp->m_reqslist;
   current = &subItems[++pos];
   while( ! current->isNil() && pos < subItems.length()-1 )
   {      
      Requirement* req = static_cast<Requirement*>(current->asInst());
      reqs.push_back( req );
      subItems.append( Item(req->cls(), req) );
      ++pos;
      current = &subItems[pos];
   }
   
}

   
void ClassModule::describe( void* instance, String& target, int , int ) const
{
   Module* mod = static_cast<Module*>(instance);
   target = "Module " + mod->name();
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

}

/* end of classmodule.cpp */
