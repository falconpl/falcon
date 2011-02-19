/*
   FALCON - The Falcon Programming Language.
   FILE: function.cpp

   Function objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/function.h>
#include <falcon/localsymbol.h>
#include <falcon/closedsymbol.h>
#include <falcon/item.h>
#include <falcon/collector.h>
#include <falcon/corefunction.h>
#include <falcon/module.h>

#include <falcon/engine.h>

namespace Falcon
{

Function::Function( const String& name, Module* module, int32 line ):
   m_name( name ),
   m_paramCount(0),
   m_gcToken( 0 ),
   m_module( module ),
   m_line( line )
{
}

Function::~Function()
{
}

void Function::module( Module* owner )
{
   //TODO Proper referencing
   m_module = owner;
}

String Function::locate() const
{
   String temp = name();
   if ( m_line != 0 )
   {
      temp.A("(").N(m_line).A(")");
   }

   if ( m_module != 0 )
   {
      if( m_module->uri().size() )
      {
         temp.A(" ").A( m_module->uri() );
      }
      else if ( m_module->name().size() )
      {
         temp.A(" [").A( m_module->name() ).A("]");
      }
   }

   return temp;
}


void Function::gcMark(int32 mark)
{
   if (m_gcToken != 0 )
   {
      m_gcToken->mark(mark);
   }
}


GCToken* Function::garbage( Collector* c )
{
   m_gcToken = c->store( Engine::instance()->functionClass(), this );
   return m_gcToken;
}

GCToken* Function::garbage()
{
   register Engine* inst = Engine::instance();
   m_gcToken = inst->collector()->store( inst->functionClass(), this );
   return m_gcToken;
}

}

/* end of function.cpp */
