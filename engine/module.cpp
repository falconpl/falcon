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

#include <falcon/atomic.h>
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
#include <falcon/textwriter.h>

#include <falcon/errors/codeerror.h>
#include <falcon/errors/genericerror.h>
#include <falcon/errors/linkerror.h>

#include <stdexcept>
#include <map>
#include <deque>
#include <list>
#include <algorithm>

#include "module_private.h"

namespace Falcon
{


Module::Private::~Private()
{
   ImportDefList::iterator id_i = m_importDefs.begin();
   while( id_i != m_importDefs.end() )
   {
      //delete *id_i;
      ++id_i;
   }

   ModReqList::iterator rl_i = m_mrlist.begin();
   while( rl_i != m_mrlist.end() )
   {
      ModRequest* mr = *rl_i;
      delete mr;
      ++rl_i;
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
   m_bNative( false ),
   m_refcount(1)
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
   m_bNative( bNative ),
   m_refcount(1)
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
   m_bNative( bNative ),
   m_refcount(1)
{
   TRACE("Creating module '%s' from %s",
      name.c_ize(), uri.c_ize() );

   _p = new Private;
}


Module::~Module()
{
   TRACE("Deleting module %s", m_name.c_ize() );
   // this is doing to do a bit of stuff; see ~Private()
   delete _p;
   TRACE("Module '%s' deletion complete", m_name.c_ize() );
}

void Module::decref()
{
   if( atomicDec(m_refcount) == 0 )
   {
      DynUnloader* unl = m_unloader;
      m_unloader = 0;
      delete this;

      if( unl != 0 ) {
         unl->unload();
      }
   }
}

uint32 Module::importCount() const
{
   return _p->m_importDefs.size();
}

ImportDef* Module::getImport( uint32 n ) const
{
   return _p->m_importDefs[n];
}


Item* Module::resolve( const String& symName )
{
   Symbol* sym = Engine::getSymbol(symName);
   try
   {
      Item* item = this->resolve(sym);
      Engine::releaseSymbol(sym);
      return item;
   }
   catch ( ... )
   {
      Engine::releaseSymbol(sym);
      throw;
   }

   return 0;
}


Item* Module::resolve( Symbol* sym )
{
   Item* value = resolveLocally(sym);

   // are we lucky?
   if( value != 0 ) {
      return value;
   }

   // no? -- try to resolve in the namespace translation map.
   length_t nspos = sym->name().rfind(".");
   String ns;
   if( nspos > 0 )
   {
      ns = sym->name().subString(0,nspos);
   }
   else {
      ns = "";
   }

   // check if there are import requests providing this namespace
   Private::NSTransMap::iterator nsti = _p->m_nsTransMap.find(ns);
   while( nsti != _p->m_nsTransMap.end() && nsti->first == ns )
   {
      Module* provider = nsti->second->modReq()->module();
      if( provider != 0 )
      {
         Item* value = provider->resolveLocally( sym );
         if( value != 0 )
         {
            m_globals.addExtern( sym, value );
            return value;
         }
      }

      // try in other modules providing the same namespace.
      ++nsti;
   }


   // no? -- try in the module space.
   if( m_modSpace != 0 )
   {
      value = m_modSpace->findExportedValue( sym->name() );
      if( value != 0 )
      {
         m_globals.addExtern( sym, value );
         return value;
      }
   }

   // still no luck? -- try in the engine
   static Engine* engine = Engine::instance();
   value = engine->getBuiltin( sym->name() );

   if( value != 0 )
   {
      m_globals.addExtern( sym, value );
      return value;
   }

   return 0;
}


Item* Module::resolveLocally(Symbol* sym)
{
   return m_globals.getValue( sym );
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

      m_attributes.gcMark(mark);

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


bool Module::addMantra(Mantra* f, bool bExport)
{
   TRACE(" Module::addMantra -- (%s(%p), %s, %d)",
            f->name().size() == 0 ? "(anon)" : f->name().c_ize(),
            f, bExport? "export" : "private", f->declaredAt() );

   GlobalsMap::Data* vd = m_globals.get( f->name() );
   if( vd != 0 && ! vd->m_bExtern )
   {
      // already defined.
      TRACE1(" Module::addMantra -- %s(%p) already defined", f->name().c_ize(), f );
      return false;
   }

   // add to the function vector so that we can account it.
   _p->m_mantras[f->name()] = f;
   f->module(this);

   // then add the required global.
   TRACE1(" Module::addMantra -- %s(%p) adding as new global", f->name().c_ize(), f );
   addGlobal( f->name(), Item( f->handler(), f ) );

   return true;
}


GlobalsMap::Data* Module::addGlobal( const String& name, const Item& value, bool bExport = true )
{
   Symbol* sym = Engine::getSymbol(name);
   GlobalsMap::Data* vd = m_globals.get( sym );
   bool decRef = vd == 0 || ! vd->m_bExtern;

   GlobalsMap::Data* data = addGlobal( sym, value, bExport );
   // if addGlobal accepted the symbol as new, it did an incref.
   if( decRef ) sym->decref();
   return data;
}


GlobalsMap::Data* Module::addGlobal( Symbol* sym, const Item& value, bool bExport )
{
   GlobalsMap::Data* vd = m_globals.get( sym );
   if( vd != 0 )
   {
      if( ! vd->m_bExtern ) {
         TRACE1(" Module::addMantra -- %s(%p) already declared", sym->name().c_ize(), sym );
         return false;
      }
      else {
         // promote.
         TRACE1(" Module::addMantra -- %s(%p) promoted from extern", sym->name().c_ize(), sym );
         vd->m_bExtern = false;
         _p->m_externals.erase( sym );
      }
   }
   else {
      TRACE1(" Module::addMantra -- %s(%p) adding", sym->name().c_ize(), sym );
      vd = m_globals.add(sym, value, bExport);
   }

   return vd;
}


bool Module::addInitClass( Class* cls, bool bExport )
{
   bool ok = addMantra( cls, false );
   if( ok )
   {
      _p->m_initList.push_back(cls);
   }

   return ok;
}


bool Module::addObject( Class* cls, bool bExport )
{
   if( cls->name().getCharAt(0) != '%')
   {
      cls->name( "%" + cls->name() );
   }
   return addInitClass(cls, bExport);
}



int32 Module::getInitCount() const
{
   return (int32) _p->m_initList.size();
}

Class* Module::getInitClass( int32 val ) const
{
   return _p->m_initList[val];
}


Variable* Module::addFunction( const String &name, ext_func_t f, bool bExport )
{
   // check if the name is free.
   GlobalsMap::Data* vd = m_globals.get( name );
   if( vd != 0 && ! vd->m_bExtern )
   {
      return 0;
   }

   // ok, the name is free; add it
   Function* extfunc = new ExtFunc( name, f, this );
   return addMantra( extfunc, bExport );
}


bool Module::addConstant( const String& name, const Item& value, bool bExport )
{
   GlobalsMap::Data* vd = m_globals.add( name, value, bExport );
   if( vd != 0 ) {
      return false;
   }

   // todo: mark the variable as constant.
   return true;
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
      if( ! rator( *m ) )
         break;
      ++iter;
   }
}


bool Module::addImplicitImport( const String& name, int32 line )
{
   if( m_globals.get(name) != 0 )
   {
      return false;
   }

   Symbol* sym = Engine::getSymbol(name);
   GlobalsMap::Data* data = m_globals.add( sym, Item(), false );
   data->m_bExtern = true;
   _p->m_externals[sym] = std::make_pair(line, 0);
   sym->decref();
   return true;
}


bool Module::addImplicitImport( Symbol* sym, int32 line )
{
   if( m_globals.get(sym) != 0 )
   {
      return false;
   }

   GlobalsMap::Data* data = m_globals.add( sym, Item(), false );
   data->m_bExtern = true;
   _p->m_externals[sym] = std::make_pair(line, 0);
   return true;
}


bool Module::addImport( ImportDef* def, Error*& error, int32 line )
{
   // start zeroing the error.
   error = 0;
   int errCount = 0;

   // first, check if the module request is compatible.
   if( ! def->sourceModule().empty() )
   {
      Private::ModReqMap::iterator pos = _p->m_mrmap.find(def->sourceModule());
      ModRequest* req;
      if( pos != _p->m_mrmap.end() )
      {
         // is this a promotion to load?
         req = pos->second;

         if( def->isLoad() )
         {
            if( req->isLoad() )
            {
               error = new LinkError( ErrorParam( e_import_already_mod, def->sr().line(), this->uri() )
                              .origin( ErrorParam::e_orig_compiler )
                              .extra(def->sourceModule() ) );
               return false;
            }
            else {
               req->promoteLoad();
            }
         }
         else {
            // link request and import definition
            req->addImportDef(def);
            def->modReq(req);
         }
      }
      else
      {
         // this is a new request.
         req = addModRequest( def->sourceModule(), def->isUri() );

         // link the definition in it
         req->addImportDef(def);
      }
   }

   // then, check if there is a clash with already defined symbols.
   int symcount = def->symbolCount();
   for( int i = 0; i < symcount; ++ i )
   {
      String name;
      def->targetSymbol( i, name );
      if( name.size() == 0 ) continue; // a bit defensive.

      if( name.getCharAt( name.length() -1 ) !=  '*' )
      {
         Symbol* imported = Engine::getSymbol(name);

         // it's a real symbol.
         if ( m_globals.get( imported ) != 0 )
         {
            imported->decref();
            LinkError* err = new LinkError( ErrorParam( e_import_already, def->sr().line(), this->uri() )
               .origin( ErrorParam::e_orig_compiler )
               .extra(name) );

            if( errCount == 0 )
            {
               error = err;
            }
            else
            {
               if( errCount == 1 )
               {
                  LinkError* err1 = new LinkError( ErrorParam( e_link_error, def->sr().line(), this->uri() )
                           .origin( ErrorParam::e_orig_compiler )
                           .extra(name) );
                  err1->appendSubError(error);
                  error = err1;
               }
               error->appendSubError( err );
            }

            ++errCount;
         }
         else
         {
            // create a global entry, and add it to the external requirements.
            m_globals.add( imported, Item(), false );
            _p->m_externals.insert(std::make_pair( imported, Private::ExtDef(line, def) ));
         }
      }
   }

   // save the definition
   _p->m_importDefs.push_back( def );

   // prepare the namespace resolution map for this definition.
   if( def->isGeneric() )
   {
      // gives symbols in the generic namespace.
      _p->m_nsTransMap.insert( std::make_pair( "", def) );
   }
   else
   {
      // Check all the namespaces provided by this request
      int symcount = def->symbolCount();
      for( int i = 0; i < symcount; ++ i )
      {
         String name;
         def->targetSymbol( i, name );

         // is this a namespace request?
         if( ! name.empty() && name.endsWith("*") )
         {
            // import * from xyz?
            if( name == "*" ) {
               name = "";
            }
            else
            {
               // let's be forgiving, check both import a.* from xyz and import a* from xyz
               if( name.endsWith(".*") )
               {
                  name = name.subString(0,name.length()-2);
               }
               else {
                  name = name.subString(0,name.length()-1);
               }

               _p->m_nsTransMap.insert( std::make_pair( name, def) );
            }
         }

         // we don't care about precise symbols, they are already declared externals.
      }
   }

   return 0;
}


ModRequest* Module::addModRequest( const String& name, bool isUri, bool isLoad )
{
   // add the module request
   ModRequest* req = new ModRequest( name, isUri, isLoad );
   _p->m_mrlist.push_back(req);
   _p->m_mrmap[name] = req;
   return req;
}


void Module::onLoad()
{
}

void Module::onModuleResolved( ModRequest* )
{
}

void Module::onImportResolved( ImportDef*, Symbol*, Item* )
{
}

void Module::onLinkComplete()
{
}

static void pushAttribs( VMContext* ctx, const AttributeMap& map )
{
   static PStep* attribStep = &Engine::instance()->stdSteps()->m_fillAttribute;

   uint32 count = map.size();
   for( uint32 i = 0; i < count; ++i )
   {
      Attribute* attrib = map.get( i );
      if( attrib->generator() != 0 ) {
         ctx->pushData( Item(Attribute::CLASS_NAME, attrib ) );
         ctx->pushCode( attribStep );
         ctx->pushCode( attrib->generator() );
      }
   }
}


void Module::startup( VMContext* ctx )
{
   static PStep* initStep = &Engine::instance()->stdSteps()->m_fillInstance;

   TRACE( "Module::startup for module %s", name().c_ize() );

   // check all the mantras...
   Private::MantraMap::iterator iter = _p->m_mantras.begin();
   while( iter != _p->m_mantras.end() )
   {
      Mantra* mantra = iter->second;
      if( mantra->isCompatibleWith(Mantra::e_c_falconclass) )
      {
         FalconClass* fcls = static_cast<FalconClass*>(mantra);
         TRACE( "Module::startup for module %s -- constructing Falcon class %s",
                  name().c_ize(), mantra->name().c_ize() );

         // can the class be constructed?
         if( ! fcls->construct(ctx) )
         {
            // no? -- use an hyper construct.
            // notice that in case of hard errors, as undefined symbols, we threw here.
            HyperClass* cls = fcls->hyperConstruct();
            _p->m_mantras[cls->name()] = cls;
            Item* icls = m_globals.get(cls->name());
            fassert(icls!=0);
            icls->setUser(cls->handler(), cls);
         }
      }

      ++iter;
   }

   int32 icount = getInitCount();
   TRACE( "Module::startup for module %s -- creating %d singletons.",
                    name().c_ize(), icount );

   if( icount != 0 )
   {
     // prepare all the required calls.

      for( int32 i = 0; i < icount; ++i )
      {
         Class* cls = getInitClass(i);
         ctx->pushCode( initStep );
         ctx->callItem( Item(cls->handler(), cls) );
      }
   }

   TRACE( "Module::startup for module %s -- preparing attributes.",
                    name().c_ize() );

   // check module attributes
   pushAttribs( ctx, attributes() );

   // check mantra attributes
   Module::Private::MantraMap::iterator mi = _p->m_mantras.begin();
   Module::Private::MantraMap::iterator me = _p->m_mantras.end();
   while( mi != me )
   {
      Mantra* mantra = mi->second;
      TRACE1( "Module::startup for module %s -- preparing attributes for %s",
                        name().c_ize(), mantra->name().c_ize() );
      pushAttribs( ctx, mantra->attributes() );
      if( mantra->isCompatibleWith( Mantra::e_c_falconclass ) )
      {
         FalconClass* cls = static_cast<FalconClass*>(mantra);
         cls->registerAttributes( ctx );
      }
      ++mi;
   }

}


bool Module::addLoad( const String& name, bool bIsUri, Error*& error, int32 line = 0 )
{
   ImportDef* id = new ImportDef;
   id->setLoad( name, bIsUri );
   return addImport(id, error, line);
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


Function* Module::getMainFunction()
{
   return m_mainFunc;
}


void Module::setMainFunction( Function* mf )
{
   m_mainFunc = mf;
   mf->setMain(true);
   mf->module(this);
   mf->name("__main__");
   addMantra( mf, false );
}


void Module::addIString( const String& iString )
{
   _p->m_istrings.insert( iString );
}


void Module::enumerateIStrings( IStringEnumerator& cb ) const
{
   Private::StringSet::iterator iter = _p->m_istrings.begin();
   while( iter != _p->m_istrings.end() )
   {
      cb(*iter);
      ++iter;
   }
}

uint32 Module::countIStrings() const
{
   return _p->m_istrings.size();
}


//=====================================================================
// render
//=====================================================================

void Module::render( TextWriter* tw, int32 depth )
{
   Private::MantraMap::iterator iter = _p->m_mantras.begin();
   attributes().render(tw, depth);

   while( iter != _p->m_mantras.end() )
   {
      Mantra* m = iter->second;
      if( m->name() != "__main__"
         && m->name() != ""
         && ! m->name().startsWith("_anon#")
         )
      {
         //keep it for later.
         m->render( tw, depth );
         tw->write("\n");
      }

      ++iter;
   }

   Function* func = getMainFunction();
   if( func != 0 )
   {
      func->renderFunctionBody(tw, 0);
   }
}

}

/* end of module.cpp */
