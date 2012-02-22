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
#include "falcon/symbol.h"
#include "falcon/itemarray.h"
#include <falcon/modrequest.h>
#include <falcon/importdef.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon
{

ClassModule::ClassModule()
{
}

ClassModule::~ClassModule()
{
}


void ClassModule::dispose( void* self ) const
{
   Module* mod = static_cast<Module*>(self);
   mod->unload();
   delete module;
}


void* ClassModule::clone( void* source ) const
{
   Module* mod = static_cast<Module*>(source);
   TRACE( "Cloning module %p (%s - %s)", mod, mod->name().c_ize(), mod->uri().c_ize() );
   
   Module* modcopy = Module(*mod);
   return modcopy;
}


void* ClassModule::createInstance() const
{
   return new Module;
}

   
void ClassModule::store( VMContext* ctx, DataWriter* stream, void* instance ) const
{
   Module* mod = static_cast<Module*>(instance);
   TRACE( "Storing module %p %s (%s - %s)", mod, 
      (mod->isNative()?"native" : "syntactic" )
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
   
   // first, write the symbols.
   {
      Module::Private::GlobalsMap& globs = mp->m_gSyms;
      progID = (int32) globs.size();
      stream->write( progID );
      Module::Private::GlobalsMap::iterator globi = globs.begin();
      progID = 0;
      while( globi != globs.end() ) {
         Symbol* sym = globi->second;
         // reset the ID here
         sym->localId( progID ); 
         stream->write( sym->name() );

         ++globi;
         ++ progID;
      }

      // also the exported
      Module::Private::GlobalsMap& exps = mp->m_gExports;
      progID = (int32) exps.size();
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
      stream->write( progID );
      Module::Private::ImportDefList::iterator idi = idlist.begin();
      progID = 0;
      while( idi != idlist.end() ) {
         ImportDef* def = *idi;
         def->id( progID++ );   
         def->store( stream );
         // save the progressive ID.
         ++idi;
      }
   }
   
   // namespace imports.
   {
      Module::Private::NSImportList& nsilist = mp->m_nsimports;
      progID = (int32) nsilist.size();
      stream->write( progID );
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
}


void ClassModule::restore( VMContext* ctx, DataReader* stream, void*& empty ) const
{
   
}

void ClassModule::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   Module* mod = static_cast<Module*>(instance);
   TRACE( "Flattening module %p %s (%s - %s)", mod, 
      (mod->isNative()?"native" : "syntactic" )
      mod->name().c_ize(), mod->uri().c_ize() );
   
   if( mod->isNative() )
   {
      // nothing to do,
      return;
   }
      
   // First, prepare to save the module ids.
   Module::Private* mp = mod->_p;
   
   subItems.reserve( 
      mp->m_gSyms.size() +
      mp->m_functions.size()+
      mp->m_classes.size() +
      mp->m_reqslist.size() +
      3
   );
   
   // First, save the symbol values.   
   {
      Module::Private::GlobalsMap& globs = mp->m_gSyms;
      Module::Private::GlobalsMap::iterator globi = globs.begin();
      while( globi != globs.end() ) {
         Symbol* sym = *globi;
         Item* value = sym->getValue( ctx );
         subItems.append( value == 0 ? Item() : *value );
         ++globi;
      }
   }
   
   {
      Module::Private::FunctionMap& funcs = mp->m_functions;
      Module::Private::FunctionMap::iterator fi = funcs.begin();
      while( fi != funcs.end() ) {
         Function* func = fi->second;
         subItems.append( Item(func) );
         ++funcs;
      }
   }
   // Push a nil as a separator
   subItems.append( Item() );
   
   {
      Module::Private::ClassMap& clss = mp->m_classes;
      Module::Private::ClassMap::iterator ci = clss.begin();
      while( ci != clss.end() ) {
         Class* cls = ci->second;
         subItems.append( Item(cls) );
         ++ci;
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
   
   // complete.
}

void ClassModule::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   
void ClassModule::describe( void* instance, String& target, int maxDepth, int maxLength ) const
{
   Module* mod = static_cast<Module*>(instance);
   target = "Module " + mod->name();
}

bool ClassModule::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   // NAME 
   
}

}

/* end of classmodule.cpp */
