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

#include <falcon/module.h>
#include <falcon/module.h>
#include <falcon/itemarray.h>
#include <falcon/globalsymbol.h>
#include <falcon/unknownsymbol.h>
#include <falcon/extfunc.h>
#include <falcon/item.h>

#include <map>

namespace Falcon {

class Module::Private
{
public:
   typedef std::map<String, Symbol*> GlobalsMap;
   GlobalsMap m_gSyms;
   GlobalsMap m_gExports;

   typedef std::map<String, UnknownSymbol*> UnknownMap;
   UnknownMap m_gImports;

   typedef std::map<String, Function*> FunctionMap;
   FunctionMap m_functions;

   typedef std::map<String, Class*> ClassMap;
   ClassMap m_classes;

   ItemArray m_globals;
   bool m_bIsStatic;

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
         if( sym->type() == Symbol::t_global_symbol )
         {
            static_cast<GlobalSymbol*>(sym)->decref();
         }
         else
         {
            delete sym;
         }
         ++iter;
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
            cli->second->module(0);
            delete cli->second;
            ++cli;
         }

         FunctionMap::iterator vi = m_functions.begin();
         while( vi != m_functions.end() )
         {
            // set the module to 0, so that we're not dec-reffed.
            vi->second->module(0);
            // then delete the function.
            delete vi->second;
            ++vi;
         }
      }
   }
};



Module::Module( const String& name, bool bIsStatic ):
      m_name( name ),
      m_bIsStatic(bIsStatic)
{
   m_uri = "internal:" + name;
   _p = new Private(bIsStatic);
}


Module::Module( const String& name, const String& uri, bool bIsStatic ):
      m_name( name ),
      m_uri(uri),
      m_bIsStatic(bIsStatic)
{
   _p = new Private(bIsStatic);
}


Module::~Module()
{
   // this is doing to do a bit of stuff; see ~Private()
   delete _p;
}


GlobalSymbol* Module::addFunction( Function* f, bool bExport )
{
   //static Engine* eng = Engine::instance();

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(f->name()) != syms.end() )
   {
      return 0;
   }

   // add a proper object in the global vector
   _p->m_globals.append( f );

   // add the symbol to the symbol table.
   GlobalSymbol* sym = new GlobalSymbol( f->name(),
         _p->m_globals.at(_p->m_globals.length()-1) );
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
   static Collector* coll = Engine::instance()->collector();
   static Class* ccls = Engine::instance()->classClass();

   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(fc->name()) != syms.end() )
   {
      return 0;
   }

   // If the module is not static, garbage-ize the class
   if( ! m_bIsStatic  )
   {
      _p->m_globals.append( FALCON_GC_STORE( coll, ccls, fc ) );
   }
   else
   {
      _p->m_globals.append( Item(ccls, fc) ); // user data
   }

   // add a proper object in the global vector
   // add the symbol to the symbol table.
   GlobalSymbol* sym = new GlobalSymbol( fc->name(),
         _p->m_globals.at(_p->m_globals.length()-1) );
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

   return sym;
}


GlobalSymbol* Module::addVariable( const String& name, const Item& value, bool bExport )
{
   // check if the name is free.
   Private::GlobalsMap& syms = _p->m_gSyms;
   if( syms.find(name) != syms.end() )
   {
      return 0;
   }

   // add a proper object in the global vector
   _p->m_globals.append( value );

   // add the symbol to the symbol table.
   GlobalSymbol* sym = new GlobalSymbol( name,
         _p->m_globals.at(_p->m_globals.length()-1) );
   syms[name] = sym;

   if( bExport )
   {
      _p->m_gExports[name] = sym;
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

void Module::enumerateImports( USymbolEnumerator& rator ) const
{
   const Private::UnknownMap& syms = _p->m_gImports;
   Private::UnknownMap::const_iterator iter = syms.begin();

   while( iter != syms.end() )
   {
      UnknownSymbol* sym = iter->second;
      if( ! rator( *sym, ++iter == syms.end()) )
         break;
   }
}

}

/* end of module.cpp */

