/*
   FALCON - The Falcon Programming Language.
   FILE: famloader.cpp

   Precompiled module deserializer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/famloader.cpp"

#include <falcon/famloader.h>
#include <falcon/modspace.h>
#include <falcon/stream.h>
#include <falcon/errors/ioerror.h>
#include <falcon/restorer.h>
#include <falcon/module.h>

namespace Falcon
{

FAMLoader::FAMLoader( ModSpace* ms ):
   m_modSpace( ms )
{}

FAMLoader::~FAMLoader()
{}
   
Module* FAMLoader::load( Stream* ins , const String& path, const String& name )
{
   char buffer[4];
   ins->read( buffer, sizeof(buffer) );

   if( buffer[0] != 'F' || buffer[1] != 'M' )
   {
      throw new IOError( ErrorParam( e_mod_not_fam, __LINE__, SRC ) 
         .origin( ErrorParam::e_orig_loader )
         .extra(path) );
   }
   else if ( buffer[2] < 4 || buffer[3] < 1 )
   {
      throw new IOError( ErrorParam( e_mod_unsupported_fam, __LINE__, SRC ) 
         .origin( ErrorParam::e_orig_loader )
         .extra(path) );
   }

   Restorer rest( m_modSpace->context() );
   // in modules, all the classes are stored by mantra.
   rest.restore( ins, m_modSpace );
   Class* handler = 0;
   void* data = 0;
   bool first = 0;
   rest.next( handler, data, first );

   fassert( handler == Engine::instance()->getMantra("Module") );

   Module* mod = static_cast<Module*>( data );
   mod->name( name );
   mod->uri( path );
   return mod;
}

}

/* end of famloader.cpp */
