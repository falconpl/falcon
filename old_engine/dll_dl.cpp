/*
   FALCON - The Falcon Programming Language.
   FILE: dll_win.cpp

   Implementation of windows specific DLL system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dll_dl.h>
#include <falcon/autocstring.h>

namespace Falcon
{

DllLoader_dl::~DllLoader_dl()
{
   close();
}

bool DllLoader_dl::open( const String &dll_name )
{
   AutoCString name( dll_name );
   if( m_module != 0 )
      if ( ! dlclose( m_module ) )
         return false;

   m_module = dlopen( name.c_str(), RTLD_NOW );
   if ( m_module == 0 )
      return false;
   return true;
}

bool DllLoader_dl::close()
{
   if ( m_module != 0 ) {
      if ( dlclose( m_module ) ) {
         m_module = 0;
         return true;
      }
   }
   return false;
}

void DllLoader_dl::assign( DllLoader_dl &other )
{
   if ( m_module != 0 )
      close();

   m_module = other.m_module;
   other.m_module = 0;
}


DllFunc DllLoader_dl::getSymbol( const String &sym_name ) const
{
   AutoCString name( sym_name );

   if ( m_module != 0 )
      return DllFunc( dlsym( m_module, name.c_str() ) );
   return DllFunc( 0 );
}

bool DllLoader_dl::isDllMark( char ch1, char ch2 )
{
   if ( ch1 == 0x7f && ch2 == 'E' ) return true;
   return false;
}

void DllLoader_dl::getErrorDescription( String &descr ) const
{
   const char *le = dlerror();
   if ( le == 0 )
      return;
   descr.bufferize( le );
}

}


/* end of dll_dl.cpp */
