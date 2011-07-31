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

#include <map>
#include <vector>

namespace Falcon {

class ModSymbol
{
public:
   Module* m_module;
   Symbol* m_symbol;
   
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
   
   ModSymbol( Module* mod, bool bLoad ):
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
   
   typedef std::vector<Module*> ModLoadOrder;
   ModLoadOrder m_loadOrder;
   
   typedef std::vector<ErrorDef*> ErrorDef;
   ModLoadOrder m_loadOrder;
};


ModSpace::ModSpace( VMachine* owner ):
   m_vm( owner ),
   _p( new Private )
{
}

ModSpace::~ModSpace()
{
   delete _p;
}
   

void ModSpace::addLinkError( int err_id, const String& modName, const Symbol* sym, const String& extra )
{
   Error* e = new LinkError( ErrorParam( err_id )
      .origin(e_orig_vm)
      .line( sym != 0 ? sym->declaredAt() : 0 )
      .module( modName != "" ? modName : "<internal>" ) );
      
   addLinkError( e );
}


Module* ModSpace::findModule( const String& local_name, bool &isLoad ) const
{
   
}


bool ModSpace::promoteLoad( const String& local_name )
{
}
   
bool ModSpace::addModule( const String& local_name, Module* mod, bool isLoad )
{
}
   
bool ModSpace::link()
{
}
   

void ModSpace::readyVM()
{
}
   
 
const Symbol* ModSpace::findExportedSymbol( const String& name, Module*& declarer ) const
{
}
   

bool ModSpace::addExportedSymbol( Module* mod, const Symbol* sym )
{

}
   
}

/* end of modspace.cpp */
