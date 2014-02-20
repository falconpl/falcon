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
#include <falcon/vmcontext.h>
#include <falcon/stderrors.h>
#include <falcon/symbol.h>
#include <falcon/itemarray.h>
#include <falcon/modrequest.h>
#include <falcon/importdef.h>
#include <falcon/mantra.h>
#include <falcon/itemdict.h>
#include <falcon/stdhandlers.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <vector>

namespace Falcon
{

   /*#
    @class Module

    @prop attributes List of attributes declared in the module.
    @prop name The logical name of the module
    @prop uri The logical URI location of the module.
    @prop globals A dictionary containing all the global values in the module.

    */

   /*#
      * @method getAttribute Module
      * @brief Gets the desired attribute, if it exists.
      * @param name The name of the required attribute
      * @return value of the require attribute
      * @raise Access error if the attribute is not found
      *
      */


     /*#
      * @method setAttribute Module
      * @brief Sets or deletes the desired attribute if it exists
      * @param name The name of the required attribute
      * @optparam value The new value for the required attribute
      *
      * If @b value is not given, then the attribute is removed, if it exists.
      *
      * If @b value is given, the value is changed or created as required.
      * In this case, if the attribute doesn't exists, it is created.
      */

        /*#
      * @method add Module
      * @brief Adds a mantra to the module.
      * @param mantra The mantra to add to the module.
      * @optparam export Whether or not to export the mantra from the module.
      *
      * If @b export is not given, then the mantra is exported by default.
      *
      * If @b export is given, the mantra is exported if export is true.
      */

   /*#
 * @method addGlobal Module
 */
   /*#
 * @method setGlobal Module
 */
   /*#
 * @method getGlobal Module
 */

namespace {

//========================================================================
// Methods
//

FALCON_DECLARE_FUNCTION(init, "name:S,uri:[S]")
FALCON_DEFINE_FUNCTION_P(init)
{
   Item* i_name = ctx->param(0);
   Item* i_uri = ctx->param(1);

   if( pCount < 1
      || ! i_name->isString()
      || (pCount > 1 && ! i_uri->isString())
      )
   {
      throw paramError(__LINE__,SRC);
   }

   Module* module;
   if( pCount > 1 )
   {
      module = new Module(*i_name->asString(), *i_uri->asString());
   }
   else
   {
      module = new Module(*i_name->asString());
   }

   ctx->returnFrame(FALCON_GC_STORE(this->methodOf(), module));
}

FALCON_DECLARE_FUNCTION(getAttribute, "name:S")
FALCON_DEFINE_FUNCTION_P1(getAttribute)
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_name = ctx->param(0);
   if( ! i_name->isString() )
   {
      ctx->raiseError(paramError(__LINE__,SRC));
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


FALCON_DECLARE_FUNCTION(setAttribute, "name:S,value:X")
FALCON_DEFINE_FUNCTION_P1(setAttribute)
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_name = ctx->param(0);
   Item* i_value = ctx->param(1);
   if( i_name == NULL || ! i_name->isString() )
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

      attr->value().copyInterlocked( *i_value );
   }

   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(addMantra, "mantra:Mantra,exp:[B]")
FALCON_DEFINE_FUNCTION_P1(addMantra)
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   static Class* clsMantra = Engine::instance()->handlers()->mantraClass();

   Item* i_mantra = ctx->param(0);
   Item* i_export = ctx->param(1);
   if( i_mantra == NULL || ! i_mantra->asClass()->isDerivedFrom(clsMantra) )
   {
      ctx->raiseError(paramError());
      return;
   }

   bool bExport = true;
   if ( i_export != NULL )
   {
      if ( ! i_export->isBoolean() )
      {
         ctx->raiseError(paramError());
         return;
      }
      bExport = i_export->asBoolean();
   }

   void* inst;
   Class* cls;
   i_mantra->forceClassInst(cls, inst);

   Mantra* mantra = static_cast<Mantra*>(inst);
   Module* module = static_cast<Module*>(self.asInst());

   module->addMantra(mantra, bExport);

   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(addGlobal, "name:S,value:X,exp:[B]")
FALCON_DEFINE_FUNCTION_P1(addGlobal)
{
   Item* i_name = ctx->param(0);
   Item* i_value = ctx->param(1);
   Item* i_export = ctx->param(2);

   if(   i_name == 0 || ! i_name->isString()
      || i_value == 0
      )
   {
      throw paramError(__LINE__, SRC);
   }

   Module* module = ctx->tself<Module*>();
   const String& name = *i_name->asString();
   bool bExport = i_export == 0 ? false : i_export->isTrue();
   bool bOk = module->addGlobal(name, *i_value, bExport ) != 0;

   ctx->returnFrame(Item().setBoolean(bOk));
}


FALCON_DECLARE_FUNCTION(setGlobal, "name:S,value:X")
FALCON_DEFINE_FUNCTION_P1(setGlobal)
{
   Item* i_name = ctx->param(0);
   Item* i_value = ctx->param(1);

   if(   i_name == 0 || ! i_name->isString()
      || i_value == 0
      )
   {
      throw paramError(__LINE__, SRC);
   }

   Module* module = ctx->tself<Module*>();
   const String& name = *i_name->asString();
   Item* value = module->resolve(name);
   if( value != 0 )
   {
      value->copyFromLocal(*i_value);
   }
   else {
      throw FALCON_SIGN_XERROR(AccessError, e_undef_sym, .extra(name));
   }

   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION(getGlobal, "name:S,dflt:[X]")
FALCON_DEFINE_FUNCTION_P1(getGlobal)
{
   Item* i_name = ctx->param(0);
   Item* i_dflt = ctx->param(1);

   if( i_name == 0 || ! i_name->isString() )
   {
      throw paramError(__LINE__, SRC);
   }

   Module* module = ctx->tself<Module*>();
   const String& name = *i_name->asString();
   Item* value = module->resolve(name);
   Item result;
   if( value != 0 )
   {
      result.copyFromRemote(*value);
   }
   else if(i_dflt != 0)
   {
      result = *i_dflt;
   }
   else {
      throw FALCON_SIGN_XERROR(AccessError, e_undef_sym, .extra(name));
   }

   ctx->returnFrame(result);
}


static void get_attributes( const Class*, const String&, void* instance, Item& value )
{
   ItemDict* dict = new ItemDict;
   Module* mod = static_cast<Module*>(instance);
   uint32 size = mod->attributes().size();
   for( uint32 i = 0; i < size; ++i ) {
      Attribute* attr = mod->attributes().get(i);
      dict->insert( FALCON_GC_HANDLE( new String(attr->name())), attr->value() );
   }
   value = FALCON_GC_HANDLE(dict);
}

static void get_globals( const Class*, const String&, void* instance, Item& value )
{
   ItemDict *globs = new ItemDict;

  class Rator: public GlobalsMap::VariableEnumerator  {
  public:
     Rator(ItemDict* g): m_globs(g) {}
     virtual ~Rator() {};
     virtual void operator() ( const Symbol* sym, Item*& value )
     {
        m_globs->insert( FALCON_GC_HANDLE(
                 new String(sym->name())), *value );
     }

  private:
     ItemDict* m_globs;
  }
  rator(globs);

  Module* mod = static_cast<Module*>(instance);
  mod->globals().enumerate(rator);
  value = FALCON_GC_HANDLE(globs);
}


static void get_uri( const Class*, const String&, void* instance, Item& value )
{
   Module* mod = static_cast<Module*>(instance);
   value = FALCON_GC_HANDLE(new String(mod->uri()));
}

static void get_name( const Class*, const String&, void* instance, Item& value )
{
   Module* mod = static_cast<Module*>(instance);
   value = FALCON_GC_HANDLE(new String(mod->name()));
}

}

ClassModule::ClassModule():
   Class("Module", FLC_CLASS_ID_MODULE)
{
   m_clearPriority = 3;

   setConstuctor( new FALCON_FUNCTION_NAME(init) );
   addMethod( new FALCON_FUNCTION_NAME(getAttribute) );
   addMethod( new FALCON_FUNCTION_NAME(setAttribute) );
   addMethod( new FALCON_FUNCTION_NAME(addMantra) );
   addMethod( new FALCON_FUNCTION_NAME(getGlobal) );
   addMethod( new FALCON_FUNCTION_NAME(setGlobal) );
   addMethod( new FALCON_FUNCTION_NAME(addGlobal) );

   addProperty("attributes", &get_attributes);
   addProperty("name", &get_name);
   addProperty("uri", &get_uri );
   addProperty("globals",&get_globals);

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
   return FALCON_CLASS_CREATE_AT_INIT;
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

   // Now store the module requests
   {
      // first, number the module requests.
      Module::Private::ModReqList& mrlist = mp->m_mrlist;
      progID = (int32) mrlist.size();
      TRACE1( "ClassModule::store -- storing %d mod requests", progID );
      stream->write( progID );
      Module::Private::ModReqList::iterator mri = mrlist.begin();
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


   // Import definition
   MESSAGE1( "Module store import definition." );
   {
      Module::Private::Externals& exts = mp->m_externals;
      progID = (int32) exts.size();
      stream->write( progID );
      TRACE1( "ClassModule::store -- storing %d externals", progID );
      Module::Private::Externals::iterator depi = exts.begin();
      progID = 0;
      while( depi != exts.end() ) {
         const Symbol* sym = depi->first;

         stream->write( sym->name() );
         // line
         stream->write( depi->second.m_line );
         ImportDef* def = depi->second.m_def;
         stream->write( def == 0 ? -1 : def->id() );
         const Symbol* srcSym = depi->second.m_srcSym;
         stream->write( srcSym == 0 ? "" : srcSym->name() );

         ++depi;
      }
   }

   MESSAGE1( "Module store namespace translations." );
   {
      Module::Private::NSTransMap& nstm = mp->m_nsTransMap;
      progID = (int32) nstm.size();
      stream->write( progID );
      TRACE1( "ClassModule::store -- storing %d namespace translations", progID );
      Module::Private::NSTransMap::iterator depi = nstm.begin();
      progID = 0;
      while( depi != nstm.end() ) {
         const String& name = depi->first;
         stream->write( name );
         ImportDef* def = depi->second;
         fassert( def != 0 );
         stream->write( def->id() );

         ++depi;
      }
   }

   MESSAGE1( "Module store attributes." );

   // store the attributes
   mod->attributes().store(stream);

   MESSAGE1( "Module store international strings." );
   {
      Module::Private::StringSet& sset = mod->_p->m_istrings;
      uint32 size = sset.size();
      stream->write( size );
      Module::Private::StringSet::iterator iter = sset.begin();
      while( sset.end() != iter )
      {
         stream->write( *iter );
         ++iter;
      }
   }

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

   // Now restore the module requests
   Module::Private::ModReqList& mrlist = mp->m_mrlist;
   Module::Private::ModReqMap& mrmap = mp->m_mrmap;

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
      Module::Private::ModReqList::iterator mri = mrlist.begin();
      while( mri != mrlist.end() )
      {
         ModRequest* req = *mri;
         uint32 count;
         stream->read( count );
         TRACE1( "ClassModule::restoreModule -- Request %d has %d imports", req->id(), count );

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
         }
         ++mri;
      }
   }

   // dependencies.
   Module::Private::Externals& exts = mp->m_externals;
   stream->read( count );
   TRACE1( "ClassModule::restoreModule -- reading %d dependencies", count );
   progID = 0;
   while( progID < count )
   {
      String sName;
      String sSrcName;
      int32 line, idDef;

      stream->read( sName );
      stream->read( line );
      stream->read( idDef );
      stream->read( sSrcName );

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

      if( sSrcName !=  "" )
      {
         exts.insert( std::make_pair(Engine::getSymbol(sName), Module::Private::ExtDef(line, idef, sSrcName ) ));
      }
      else {
         exts.insert( std::make_pair(Engine::getSymbol(sName), Module::Private::ExtDef(line, idef ) ));
      }

      TRACE2( "ClassModule::restoreModule -- restored dependency %d: %s idef:%d",
               progID, sName.c_ize(), idDef );

      ++progID;
   }

   // translations.
   Module::Private::NSTransMap& nstm = mp->m_nsTransMap;
   stream->read( count );
   TRACE1( "ClassModule::restoreModule -- reading %d namespace translations", count );
   progID = 0;
   while( progID < count )
   {
      String sName;
      int32 idDef;

      stream->read( sName );
      stream->read( idDef );

      ImportDef* idef = 0;

      if( idDef < 0 || idDef >= (int32) idlist.size() )
      {
         throw new IOError( ErrorParam( e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_loader )
            .extra(String("ImportDef out of range dependency ").N(progID) )
            );
      }

      idef = idlist[idDef];
      nstm.insert(std::make_pair(sName,idef));

      TRACE2( "ClassModule::restoreModule -- restored translation %d: %s idef:%d",
               progID, sName.c_ize(), idDef );

      ++progID;
   }


   MESSAGE1( "Module restore -- attributes" );

   // restore the attributes
   mod->attributes().restore(stream);

   MESSAGE1( "Module restore -- international strings." );
   {
      Module::Private::StringSet& sset = mod->_p->m_istrings;
      uint32 size = 0;
      stream->read( size );
      for( uint32 i = 0; i < size; ++i )
      {
         String temp;
         stream->read( temp );
         sset.insert(temp);
      }
   }

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
      mod->globals().size()*3 +
      mp->m_mantras.size()+
      mod->attributes().size() * 2 +
      4
   );

   // save all the global variables
   mod->globals().flatten(ctx, subItems);
   TRACE( "ClassModule::flatten -- stored %d variables", (uint32) subItems.length() / 3 );
   // Push a nil as a separator
   subItems.append(Item());

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
   TRACE( "ClassModule::unflatten -- restored %d globals", pos/3 );
   ++pos; // skip the nil spearator after globals.

   const Item* current = &subItems[pos];
   while( ! current->isNil() && pos < subItems.length()-2 )
   {
      Mantra* mantra = static_cast<Mantra*>(current->asInst());
      TRACE1( "ClassModule::unflatten -- restoring mantra %s ", mantra->name().c_ize() );

      if( mantra->name() == "__main__" )
      {
         mod->m_mainFunc = static_cast<Function*>(mantra);
         mod->m_mainFunc->setMain(true);
      }

      mp->m_mantras[mantra->name()] = mantra;
      mantra->module( mod );
      // no need to store the mantra in globals:
      // the globals already unflattened and mantras are in place.

      ++pos;
      current = &subItems[pos];
   }

   TRACE( "ClassModule::unflatten -- restored mantras, at position %d", pos );

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
   target = "Module " + mod->name() +  " (" + mod->uri() + ")";
}

void ClassModule::gcMarkInstance( void* instance, uint32 mark ) const
{
   static_cast<Module*>(instance)->gcMark(mark);
}

bool ClassModule::gcCheckInstance( void* instance, uint32 mark ) const
{
   return static_cast<Module*>(instance)->currentMark() >= mark;
}

}

/* end of classmodule.cpp */
