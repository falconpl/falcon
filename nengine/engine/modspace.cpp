/*
   FALCON - The Falcon Programming Language.
   FILE: modspace.cpp

   Module space for the Falcon virtual machine
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 14:27:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/modspace.h>
#include <falcon/linkerror.h>
#include <falcon/error.h>
#include <falcon/symbol.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/modloader.h>

#include <map>
#include <vector>

namespace Falcon {

class ModSymbol
{
public:
   Module* m_module;
   Symbol* m_symbol;
   
   ModSymbol():
      m_module( 0 ),
      m_symbol( 0 )
   {}   
   
   ModSymbol( Module* mod, Symbol* sym ):
      m_module( mod ),
      m_symbol( sym )
   {}
};


class ModuleLoadMode
{
public:
   Module* m_module;
   bool m_bLoad;
   
   ModuleLoadMode():
      m_module(0),
      m_bLoad( false )
   {}
   
   ModuleLoadMode( Module* mod, bool bLoad ):
      m_module( mod ),
      m_bLoad( bLoad )
   {}
};


class ModSpace::Private
{
public:
   typedef std::map<String, ModSymbol> SymbolMap;
   SymbolMap m_syms;
   
   typedef std::map<String, ModuleLoadMode> ModMap;
   ModMap m_mods;
   
   typedef std::vector<ModuleLoadMode*> ModLoadOrder;
   ModLoadOrder m_loadOrder;
   
   typedef std::vector<Error*> ErrorList;
   ErrorList m_errors;
   
   Private() {}
   
   ~Private() 
   {
      clearErrors();
      
      ModMap::iterator iter = m_mods.begin();
      while( iter != m_mods.end() )
      {
         delete iter->second.m_module;
         ++iter;
      }
   }
   
   void clearErrors()
   {
      ErrorList::iterator iter = m_errors.begin();
      while( iter != m_errors.end() )
      {
         (*iter)->decref();
         ++iter;
      }
   }
};

//==============================================================
// Main class
//

ModSpace::ModSpace( VMachine* owner ):
   m_vm( owner ),
   _p( new Private )
{
   m_loader = new ModLoader;
}

ModSpace::~ModSpace()
{
   delete m_loader;
   delete _p;
}
   

void ModSpace::addLinkError( int err_id, const String& modName, const Symbol* sym, const String& extra )
{
   Error* e = new LinkError( ErrorParam( err_id )
      .origin(ErrorParam::e_orig_linker)
      .line( sym != 0 ? sym->declaredAt() : 0 )
      .extra( extra )
      .module( modName != "" ? modName : "<internal>" ) );
      
   addLinkError( e );
   e->decref();
}


void ModSpace::addLinkError( Error* e )
{
   _p->m_errors.push_back( e );
   e->incref();
}


Error* ModSpace::makeError() const
{
   Error* e = new LinkError( ErrorParam( e_link_error, __LINE__, SRC )
      .origin(ErrorParam::e_orig_linker) );
   
   Private::ErrorList::iterator iter = _p->m_errors.begin();
   while( iter != _p->m_errors.end() )
   {
      e->appendSubError( *iter );
      ++iter;
   }
   
   _p->clearErrors();
   return e;
}
   

Module* ModSpace::findModule( const String& local_name, bool &isLoad ) const
{
   Private::ModMap::iterator pos = _p->m_mods.find( local_name );
   if( pos == _p->m_mods.end() )
   {
      return 0;
   }
   
   isLoad = pos->second.m_bLoad;
   return pos->second.m_module;
}


bool ModSpace::promoteLoad( const String& local_name )
{
   Private::ModMap::iterator pos = _p->m_mods.find( local_name );
   if( pos == _p->m_mods.end() )
   {
      return false;
   }
   
   pos->second.m_bLoad = true;
   exportSymbols( pos->second.m_module );
   return true;
}

   
bool ModSpace::addModule( Module* mod, bool isLoad, bool bExcludeMain )
{
   Private::ModMap::iterator pos = _p->m_mods.find( mod->name() );
   if( pos != _p->m_mods.end() )
   {
      return false;
   }
   
   ModuleLoadMode* modload = &_p->m_mods[ mod->name() ];
   modload->m_bLoad = isLoad;
   modload->m_module = mod;
   
   if( ! bExcludeMain )
   {
      _p->m_loadOrder.push_back( modload );      
   }
      
   if ( isLoad )
   {
      exportSymbols( mod );
   }
      
   // mod->resolveDeps...
   return true;
}

   
bool ModSpace::link( bool bThrowOnError )
{   
   Private::ModLoadOrder& mods = _p->m_loadOrder;
   
   Private::ModLoadOrder::const_iterator imods = mods.begin();
   while( imods != mods.end() )
   {
      ModuleLoadMode* lm = *imods;
      Module* mod = lm->m_module;
      
      mod->passiveLink( this );
      ++imods;
   }
   
   if( bThrowOnError && ! _p->m_errors.empty() )
   {
      throw makeError();
   }
   
   return _p->m_errors.empty();
}


void ModSpace::readyVM()
{
   // TODO: set in the vm.
   
   Private::ModLoadOrder& mods = _p->m_loadOrder;
   
   VMContext* mainContext = m_vm->currentContext();
   Private::ModLoadOrder::const_reverse_iterator rimods = mods.rbegin();
   while( rimods != mods.rend() )
   {
      const Module* mod = (*rimods)->m_module;
      // TODO -- invoke the init methods.
      
      // TODO -- specific space for Main.
      Function* main = mod->getFunction("__main__");
      if( main != 0 )
      {
         mainContext->call( main, 0 );
      }
            
      ++rimods;
   }
}

 
Symbol* ModSpace::findExportedSymbol( const String& name, Module*& declarer ) const
{
   Private::SymbolMap::const_iterator pos = _p->m_syms.find( name );
   
   // found in?
   if( pos != _p->m_syms.end() )
   {
      declarer = pos->second.m_module;
      return pos->second.m_symbol;
   }
   
   return 0;
}
   

bool ModSpace::addExportedSymbol( Module* mod, Symbol* sym, bool bAddError )
{
   Private::SymbolMap::iterator pos = _p->m_syms.find( sym->name() );
   
   // Already in?
   if( pos != _p->m_syms.end() )
   {
      // Shall we add some error marker?
      if( bAddError )
      {
         // Else, describe the problem.
         String extra;
         ModSymbol& ms = pos->second;
         if( ms.m_module != 0 )
         {
            extra = "in " + ms.m_module->name();
            if( ms.m_symbol->declaredAt() != 0 )
            {
               extra.A(":").N( ms.m_symbol->declaredAt() );
            }
         }
         else
         {
            extra += "internal symbol";
         }
         
         addLinkError( e_already_def, mod->name(), sym, extra );
      }
      
      return false;
   }
   
   // add the entry
   _p->m_syms[sym->name()] = ModSymbol( mod, sym );
   return true;
}
   
void ModSpace::exportSymbols( Module* mod )
{
   class Rator: public Module::SymbolEnumerator
   {
   public:
      Rator( ModSpace* owner, Module* mod ):
         m_owner( owner ),
         m_module( mod )
      {}

      virtual bool operator()( const Symbol& sym, bool )
      {
         m_owner->addExportedSymbol( m_module, const_cast<Symbol*>(&sym), true );
         return true;
      }

   private:
      ModSpace* m_owner;
      Module* m_module;
   };

   Rator rator( this, mod );
   mod->enumerateExports( rator );
}
}

/* end of modspace.cpp */
