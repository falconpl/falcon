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
#include <falcon/mt.h>

#include "modmap_private.h"

#include <map>

namespace Falcon 
{

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
   {
      m_module->unload();
   }
}


void ModMap::add( Module* mod, t_loadMode im, bool bown )
{
   Entry* ne = new Entry( mod, im, bown );
   _p->m_mtx.lock();   
   _p->m_modMap[mod] = ne;
   _p->m_modsByName[mod->name()] = ne;
   _p->m_modsByPath[mod->uri()] = ne;
   _p->m_mtx.unlock();   
}


void ModMap::remove( Module* mod )
{
   _p->m_mtx.lock();   
   _p->m_modsByName.erase( mod->name() );
   _p->m_modsByPath.erase( mod->uri() );
   
   Private::ModEntryMap::iterator pos = _p->m_modMap.find( mod );
   if ( pos != _p->m_modMap.end() )
   {
      Entry* entry = pos->second;
      _p->m_modMap.erase( pos );
      _p->m_mtx.unlock();   
      delete entry;
   }
   else
   {
      _p->m_mtx.unlock();   
   }
}


ModMap::Entry* ModMap::findByURI( const String& path ) const
{
   _p->m_mtx.lock();   
   Private::NameEntryMap::iterator pos = _p->m_modsByName.find( path );
   if( pos != _p->m_modsByName.end() )
   {
      Entry* e = pos->second;
      _p->m_mtx.unlock();   
      return e;
   }
   _p->m_mtx.unlock();   

   return 0;
}


ModMap::Entry* ModMap::findByName( const String& name ) const
{
   _p->m_mtx.lock();   
   Private::NameEntryMap::iterator pos = _p->m_modsByName.find( name );
   if( pos != _p->m_modsByName.end() )
   {
      Entry* e = pos->second;
      _p->m_mtx.unlock();   
      return e;
   }
   _p->m_mtx.unlock();
   return 0;
}


ModMap::Entry* ModMap::findByModule( Module* mod ) const
{
   _p->m_mtx.lock();   
   Private::ModEntryMap::iterator pos = _p->m_modMap.find( mod );
   if( pos != _p->m_modMap.end() )
   {
      Entry* e = pos->second;
      _p->m_mtx.unlock();
      return e;
   }
   _p->m_mtx.unlock();
   return 0;
}


void ModMap::enumerate( ModMap::EntryEnumerator& rator ) const
{
   _p->m_mtx.lock();
   Private::ModEntryMap::iterator iter = _p->m_modMap.begin();
   while( iter != _p->m_modMap.end() )
   {
      // if we own the module, it will be delete as well
      Entry* e = iter->second;
      ++iter;
      _p->m_mtx.unlock();
      rator( e ); 
      _p->m_mtx.lock();      
   }
   _p->m_mtx.unlock();
}

bool ModMap::empty() const
{
   _p->m_mtx.lock();
   bool bResult = _p->m_modMap.empty();
   _p->m_mtx.unlock();
   return bResult;
}

}

/* end of modmap.cpp */
