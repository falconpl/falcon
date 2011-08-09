/*
   FALCON - The Falcon Programming Language.
   FILE: modmap.cpp

   A simple class orderly guarding modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/modmap.h>
#include <falcon/module.h>

#include <map>

namespace Falcon 
{

class ModMap::Private
{
public:   
   typedef std::map<Module*,ModMap::Entry*> ModEntryMap;
   typedef std::map<String,ModMap::Entry*> NameEntryMap;
   
   ModEntryMap m_modMap;
   NameEntryMap m_modsByName;
   NameEntryMap m_modsByPath;   
   
   Private() {};
   
   ~Private()
   {
      // for all the modules
      ModEntryMap::iterator iter = m_modMap.begin();
      while( iter != m_modMap.end() )
      {
         // if we own the module, it will be delete as well
         delete iter->second; 
         ++iter;
      }
   }
};


ModMap::ModMap():
   _p( new Private) 
{}

ModMap::~ModMap() 
{
   delete _p;
}
      
ModMap::Entry::~Entry()
{
   if( m_bOwn )
      delete m_module;
}


void ModMap::add( Module* mod, t_importMode im, bool bown )
{
   Entry* ne = new Entry( mod, im, bown );
   _p->m_modMap[mod] = ne;
   _p->m_modsByName[mod->name()] = ne;
   _p->m_modsByPath[mod->uri()] = ne;
}


void ModMap::remove( Module* mod )
{
   _p->m_modsByName.erase( mod->name() );
   _p->m_modsByPath.erase( mod->uri() );
   
   Private::ModEntryMap::iterator pos = _p->m_modMap.find( mod );
   if ( pos != _p->m_modMap.end() )
   {
      Entry* entry = pos->second;
      _p->m_modMap.erase( pos );
      delete entry;
   }
}


ModMap::Entry* ModMap::findByURI( const String& path ) const
{
   Private::NameEntryMap::iterator pos = _p->m_modsByName.find( path );
   if( pos != _p->m_modsByName.end() )
   {
      return pos->second;
   }
   return 0;
}


ModMap::Entry* ModMap::findByName( const String& name ) const
{
   Private::NameEntryMap::iterator pos = _p->m_modsByName.find( name );
   if( pos != _p->m_modsByName.end() )
   {
      return pos->second;
   }
   return 0;
}


ModMap::Entry* ModMap::findByModule( Module* mod ) const
{
   Private::ModEntryMap::iterator pos = _p->m_modMap.find( mod );
   if( pos != _p->m_modMap.end() )
   {
      return pos->second;
   }
   return 0;
}


void ModMap::enumerate( ModMap::EntryEnumerator& rator ) const
{
   Private::ModEntryMap::iterator iter = _p->m_modMap.begin();
   while( iter != _p->m_modMap.end() )
   {
      // if we own the module, it will be delete as well
      rator(iter->second); 
      ++iter;
   }
}

 
}

/* end of modmap.cpp */
