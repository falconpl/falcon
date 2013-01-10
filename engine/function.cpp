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
#include <falcon/closure.h>
#include <falcon/callframe.h>

#include <falcon/engine.h>
#include <falcon/errors/paramerror.h>

#include <map>

namespace Falcon
{

Function::EtaSetter Function::eta;

Function::Function( const String& name, Module* module, int32 line ):
   Mantra( name, module, line ),
   m_methodOf( 0 ),
   m_bEta(false)
{
   m_category = e_c_function;
}

Function::~Function()
{
}


Class* Function::handler() const
{
   static Class* cls = Engine::instance()->functionClass();   
   return cls;
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


Error* Function::paramError(int line, const char* place ) const
{
   String placeName = place == 0 ? (m_module == 0 ? "" : m_module->name() ) : place;
   placeName.bufferize();
   return new ParamError(
           ErrorParam(e_inv_params, line == 0 ? m_sr.line(): line, placeName)
           .extra(m_signature) );
   
}

}

/* end of function.cpp */
