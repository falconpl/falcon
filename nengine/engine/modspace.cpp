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
#include <falcon/mt.h>
#include <falcon/modgroup.h>

#include <map>

namespace Falcon {


class ModSpace::Private
{
public:
   
   Mutex m_mtx;
   typedef std::map<String,String> TokenMap;
   TokenMap m_tokens;
   
   Private() {}
   
   ~Private() 
   {     
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


bool ModSpace::addSymbolToken( const String& name, const String& modName, String& declarer )
{
   _p->m_mtx.lock();
   Private::TokenMap::iterator iter = _p->m_tokens.find( name );
   if( iter != _p->m_tokens.end() )
   {
      declarer = iter->second;
      _p->m_mtx.unlock();
      return false;
   }
   
   _p->m_tokens[name] = modName;
   _p->m_mtx.unlock();
   return true;
}

void ModSpace::removeSymbolToken( const String& name, const String& modName )
{
   _p->m_mtx.lock();
   Private::TokenMap::iterator iter = _p->m_tokens.find( name );
   if( iter != _p->m_tokens.end() )
   {
      if( iter->second == modName )
      {
      _p->m_tokens.erase( iter );
      }
   }
   _p->m_mtx.unlock();
}


Symbol* ModSpace::findExportedSymbol( const String& name ) const
{
   SymbolMap::Entry* entry = symbols().find( name );
   if( entry == 0 )
   {
      return 0;
   }
   return entry->symbol();
}


Error* ModSpace::add( Module* mod, t_loadMode lm, VMContext* ctx )
{
   //static Collector* coll = Engine::instance()->collector();
   
   ModGroup* mg = new ModGroup( this );
   if ( mg->add( mod, lm ) )
   {
      if( mg->link() )
      {
         mg->readyVM( ctx );
         if( mg->modules().empty() )
         {
            delete mg;
         }
         else
         {
            //TODO
            //FALCON_GC_STORE( coll, clsModGroup, mg );
         }
         return 0;
      }
   }
   
   // we had some error.
   Error* err = mg->makeError();
   delete mg;
   return err;
}
   

/*
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
*/


}

/* end of modspace.cpp */
