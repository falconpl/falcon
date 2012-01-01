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

#include <falcon/classes/classfunction.h>

#include <falcon/function.h>
#include <falcon/symbol.h>
#include <falcon/item.h>
#include <falcon/collector.h>
#include <falcon/module.h>

#include <falcon/engine.h>

#include <falcon/errors/paramerror.h>

namespace Falcon
{

Function::EtaSetter Function::eta;


Function::Function( const String& name, Module* module, int32 line ):
   m_name( name ),
   m_paramCount(0),
   m_lastGCMark( 0 ),
   m_module( module ),
   m_methodOf( 0 ),
   m_line( line ),
   m_bEta(false)
{}

Function::~Function()
{
}


void Function::module( Module* owner )
{
   m_module = owner;
}


bool Function::parseDescription( const String& params )
{
   m_signature = "";
   
   length_t ppos = 0;
   char_t chr;
   while( (chr = params.getCharAt(ppos)) == '&' )
   {
      setEta(true);
      ++ppos;
   }
      
   if ( ppos >= params.length() )
   {
      return true;
   }
   
   length_t pos;
   do 
   {
      pos = params.find( ',', ppos );   
      String param = params.subString(ppos, pos);
      param.trim();
      length_t pColon = param.find( ":" );
      if( pColon == String::npos || pColon == 0 )
      {
         return false;
      }
      else
      {
         String pname = param.subString(0,pColon); pname.trim();
         String psig = param.subString(pColon+1); psig.trim();
         
         addParam( pname );
         if( m_signature.size() > 0 )
         {
            m_signature += ",";
         }
         m_signature += psig;
      }
      
      ppos = pos+1;      
   }
   while( pos != String::npos );
   
   return true;
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
   placeName.bufferize();
   return new ParamError(
           ErrorParam(e_inv_params, line == 0 ? m_line: line, placeName)
           .extra(m_signature) );
   
}

void Function::gcMark( uint32 mark )
{
   m_lastGCMark = mark;
   // TODO: Mark the references.
}


bool Function::gcCheck( uint32 mark )
{
   // if we're residing in a module,
   // ...let the module destroy us when it goes out of mark.
   if( m_lastGCMark < mark && m_module != 0 )
   {
      delete this;
      return false;
   }

   return true;
}

}

/* end of function.cpp */
