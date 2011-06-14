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

#include <falcon/error_messages.h>
#include <falcon/paramerror.h>

namespace Falcon
{

Function::EtaSetter Function::eta;
Function::DetermSetter Function::determ;


Function::Function( const String& name, Module* module, int32 line ):
   m_name( name ),
   m_paramCount(0),
   m_gcToken( 0 ),
   m_module( module ),
   m_line( line ),
   m_bDeterm(false),
   m_bEta(false)
{
}

Function::~Function()
{
   if ( m_module != 0 )
   {
      //TODO: Properly decreference the module.
      //m_module->decref();
   }
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
   static Class* fclass = Engine::instance()->functionClass();
   m_gcToken = c->store( fclass, this );
   return m_gcToken;
}

GCToken* Function::garbage()
{
   register Engine* inst = Engine::instance();
   m_gcToken = inst->collector()->store( inst->functionClass(), this );
   return m_gcToken;
}


Error* Function::paramError(int line, const char* place ) const
{
   String placeName = place == 0 ? (m_module == 0 ? "" : m_module->name() ) : place;
   return new ParamError(
           ErrorParam(e_inv_params, line == 0 ? m_line: line, placeName)
           .extra(m_signature) );
   
}

}

/* end of function.cpp */
