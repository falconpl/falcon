/*
   FALCON - The Falcon Programming Language.
   FILE: dll_mac.cpp

   Implementation of darwin specific DLL system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai:
   Modified for darwin by: Francesco Guerra
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dll_mac.h>
#include <falcon/autocstring.h>

namespace Falcon
{

DllLoader_Mac::~DllLoader_Mac()
{
   close();
}

bool DllLoader_Mac::open( const String &dll_name )
{
   if( m_module != 0 )
      if ( ! dlclose( m_module ) )
         return false;
   
   AutoCString name( dll_name );
   m_module = dlopen( name.c_str(), RTLD_NOW );
   if ( m_module == 0 )
      return false;
   return true;
}

bool DllLoader_Mac::close()
{
   if ( m_module != 0 ) {
      if ( dlclose( m_module ) ) {
         m_module = 0;
         return true;
      }
   }
   return false;
}

void DllLoader_Mac::assign( DllLoader_Mac &other )
{
   if ( m_module != 0 )
      close();

   m_module = other.m_module;
   other.m_module = 0;
}


DllFunc DllLoader_Mac::getSymbol( const String &sym_name ) const
{
   AutoCString name(sym_name);
   if ( m_module != 0 )
      return DllFunc( dlsym( m_module, name.c_str() ) );
   return DllFunc( 0 );
}

bool DllLoader_Mac::isDllMark( unsigned char ch1, unsigned char ch2 )
{
   if ( ch1 == 0xfe && ch2 == 0xed || ch1 == 0xca && ch2 == 0xfe ) return true;
   // Magic for Mach-O and Mach-O Fat binaries
   return false;
}

void DllLoader_Mac::getErrorDescription( String &descr ) const
{
   const char *le = dlerror();
   if ( le == 0 )
      return;
   descr.bufferize( le );
}

}


/* end of dll_mac.cpp */
