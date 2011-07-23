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
#include <falcon/classfunction.h>
#include <falcon/module.h>

#include <falcon/engine.h>

#include <falcon/error.h>
#include <falcon/paramerror.h>

namespace Falcon
{

Function::EtaSetter Function::eta;
Function::DetermSetter Function::determ;


Function::Function( const String& name, Module* module, int32 line ):
   m_name( name ),
   m_paramCount(0),
   m_lastGCMark( 0 ),
   m_module( module ),
   m_methodOf( 0 ),
   m_line( line ),
   m_bDeterm(false),
   m_bEta(false)
{
   if ( module != 0 )
   {
      module->rc.inc();
   }
}

Function::~Function()
{
   if ( m_module != 0 )
   {
      m_module->rc.dec();
   }
}


void Function::module( Module* owner )
{
   if( owner != 0 )
   {
      owner->rc.inc();
   }

   if ( m_module != 0 )
   {
      m_module->rc.dec();
   }

   m_module = owner;
}


String Function::locate() const
{
   String temp = name();
   if ( m_line != 0 )
   {
      temp.A("(").N(m_line).A(")");
   }

   Module* mod = m_module;

   if( mod == 0 && m_methodOf != 0 )
   {
      mod = m_methodOf->module();
   }


   if ( mod != 0 )
   {
      if( mod->uri().size() )
      {
         temp.A(" ").A( mod->uri() );
      }
      else if ( mod->name().size() )
      {
         temp.A(" [").A( mod->name() ).A("]");
      }
   }

   return temp;
}

Error* Function::paramError(int line, const char* place ) const
{
   String placeName = place == 0 ? (m_module == 0 ? "" : m_module->name() ) : place;
   return new ParamError(
           ErrorParam(e_inv_params, line == 0 ? m_line: line, placeName)
           .extra(m_signature) );
   
}

void Function::gcMark( uint32 mark )
{
   m_lastGCMark = mark;
}


bool Function::gcCheck( uint32 mark )
{
   if( m_lastGCMark < mark )
   {
      delete this;
      return false;
   }

   return true;
}

}

/* end of function.cpp */
