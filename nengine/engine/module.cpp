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

#include <falcon/trace.h>
#include <falcon/module.h>
#include <falcon/itemarray.h>
#include <falcon/globalsymbol.h>
#include <falcon/unknownsymbol.h>
#include <falcon/extfunc.h>
#include <falcon/item.h>

#include <falcon/inheritance.h>

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
      virtual void onSymbolLoaded( Module* mod, Symbol* sym ) = 0;
   };

   class WaitingSym: public WaitingDep
   {
   public:
      UnknownSymbol* m_symbol;

      WaitingSym( UnknownSymbol* sym ):
         m_symbol( sym )
      {}

      virtual void onSymbolLoaded( Module*, Symbol* sym )
      {
         m_symbol->define( sym );
      }
   };

   class WaitingInherit: public WaitingDep
   {
   public:
      Inheritance* m_inh;

      WaitingInherit( Inheritance* inh ):
         m_inh( inh )
      {}

      virtual void onSymbolLoaded( Module*, Symbol* sym )
      {
         //TODO -- account error if not a class or not retriveable
         Item value;
         if( sym->retrieve( value, 0 ) )
         {
            //...
         }

      }
   };

   class Dependency
   {
   public:
      String m_remoteName;
      
      typedef std::deque<WaitingDep*> WaitList;
      WaitList m_waiting;

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
      }
   };

   /** Record keeping track of needed modules (eventually already defined). */
   class Requirement
   {
   public:
      String m_uri;
      bool m_bIsUri;
      bool m_bIsLoad;
      Module* m_module;

      typedef std::map<String, Dependency*> DepMap;
      DepMap m_deps;

      Requirement( const String& name, bool bIsLoad, bool bIsUri=false, Module* mod = 0 ):
         m_uri( name ),
         m_bIsUri( bIsUri ),
         m_bIsLoad( bIsLoad ),
         m_module(mod)
      {}

      ~Requirement()
      {
         DepMap::iterator dep_i = m_deps.begin();
         while( dep_i != m_deps.end() )
         {
            delete dep_i->second;
            ++dep_i;
         }
      }
   };

   typedef std::map<String, Requirement*> ReqMap;
   ReqMap m_reqs;

   //============================================
   // Main data
   //============================================
   typedef std::map<String, Symbol*> GlobalsMap;
   GlobalsMap m_gSyms;
   GlobalsMap m_gExports;

   typedef std::map<String, Function*> FunctionMap;
   FunctionMap m_functions;

   typedef std::map<String, Class*> ClassMap;
   ClassMap m_classes;

   bool m_bIsStatic;

   ItemArray m_staticdData;


   Private( bool bIsStatic ):
      m_bIsStatic( bIsStatic )
   {}

   ~Private()
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

      // ... But in case of dynamic modules, we're destroyed only after all our
      //     functions are destroyed.
      if( m_bIsStatic )
      {
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

      }
   }

   Dependency* getDep( const String& source, bool bIsUri, const String& name )
   {
      Requirement* req;
      ReqMap::iterator ireq = m_reqs.find( name );
      if( ireq != m_reqs.end() )
      {
         // already loaded?
         req = ireq->second;
         // it is legal to import symbols even from loaded modules.
      }
      else
      {
         req = new Requirement(source, false, bIsUri );
         m_reqs[name] = req;
      }

      Dependency* dep;
      Requirement::DepMap::iterator idep = req->m_deps.find( name );
      if( idep != req->m_deps.end() )
      {
         dep = idep->second;
      }
      else
      {
         dep = new Private::Dependency( name );
         req->m_deps[name] = dep;
      }

      return dep;
   }
};



Module::Module( const String& name, bool bIsStatic ):
   m_name( name ),
   m_bIsStatic(bIsStatic),
   m_lastGCMark(0)
{
   TRACE("Creating internal module '%s'%s",
      name.c_ize(), bIsStatic ? " (static)" : " (dynamic)" );
   m_uri = "internal:" + name;
   _p = new Private(bIsStatic);
}


Module::Module( const String& name, const String& uri, bool bIsStatic ):
   m_name( name ),
   m_uri(uri),
   m_bIsStatic(bIsStatic),
   m_lastGCMark(0)
{
   TRACE("Creating module '%s' from %s%s",
      name.c_ize(), uri.c_ize(),
      bIsStatic ? " (static)" : " (dynamic)" );
   
   _p = new Private(bIsStatic);
}


Module::~Module()
{
   TRACE("Deleting module %s", m_name.c_ize() );

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
   if( m_bIsStatic || ia.empty() )
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

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(f->name()) != syms.end() )
   {
      return 0;
   }

   // add the symbol to the symbol table.
   GlobalSymbol* sym = new GlobalSymbol( f->name(), f );
   syms[f->name()] = sym;

   // if the module is dynamic, we want the GC to mark us when we generate the item.
   if( ! m_bIsStatic )
   {
      sym->value().garbage();
   }

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
      gsym->value() = f;
   }
}


GlobalSymbol* Module::addFunction( const String &name, ext_func_t f, bool bExport )
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


GlobalSymbol* Module::addVariable( const String& name, bool bExport )
{
   return addVariable( name, Item(), bExport );
}


void Module::addClass( GlobalSymbol* gsym, Class* fc, bool )
{
   static Class* ccls = Engine::instance()->classClass();

   // finally add to the function vecotr so that we can account it.
   _p->m_classes[fc->name()] = fc;
   fc->module(this);

   if(gsym)
   {
      gsym->value().setUser( ccls, fc );
   }
}


GlobalSymbol* Module::addClass( Class* fc, bool, bool bExport )
{
   static Class* ccls = Engine::instance()->classClass();

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(fc->name()) != syms.end() )
   {
      return 0;
   }

   // add a proper object in the global vector
   // add the symbol to the symbol table.
   GlobalSymbol* sym = new GlobalSymbol( fc->name(), Item(ccls, fc) );
   syms[fc->name()] = sym;

   // If the module is not static, garbage-ize the class
   if( ! m_bIsStatic  )
   {
      sym->value().garbage();
   }

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
   Private::GlobalsMap& syms = _p->m_gSyms;
   Private::GlobalsMap::iterator pos = syms.find(name);
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
   const Private::GlobalsMap& syms = _p->m_gExports;
   Private::GlobalsMap::const_iterator iter = syms.begin();

   while( iter != syms.end() )
   {
      Symbol* sym = iter->second;
      if( ! rator( *sym, ++iter == syms.end()) )
         break;
   }
}

bool Module::addLoad( const String& name, bool bIsUri )
{
   // do we have the recor?
   Private::ReqMap::iterator ireq = _p->m_reqs.find( name );
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

   // add a new record
   Private::Requirement* r = new Private::Requirement(name, true, bIsUri );
   _p->m_reqs[ name ] = r;
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

   Private::Dependency* dep = _p->getDep( "", false, name );
   UnknownSymbol* usym = new UnknownSymbol(name);
   // Record the fact that we have to save transform an unknown symbol...
   dep->m_waiting.push_back( new Private::WaitingSym( usym ) );
   // ... and save the dependency.
   _p->m_gSyms[name] = usym;

   return usym;
}


Symbol* Module::addExport( const String& name )
{
   Symbol* sym;
   
   // We can't be called if the symbol is alredy declared elsewhere.
   Private::GlobalsMap::const_iterator iter = _p->m_gSyms.find( name );
   if( iter != _p->m_gSyms.end() )
   {
      sym = iter->second;
      
      // ... and save the dependency.
      _p->m_gExports[name] = sym;          
   }
   else
   {
      sym = addVariable( name, true );
   }

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

}

/* end of module.cpp */

