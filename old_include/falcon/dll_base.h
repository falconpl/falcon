/*
   FALCON - The Falcon Programming Language.
   FILE: flc_dll.h

   Base class for Dynamic load system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_DLL_BASE_H
#define flc_DLL_BASE_H

#include <falcon/string.h>

namespace Falcon
{

class DllFunc
{
   void *m_data;

public:
   DllFunc( void *data ):
      m_data(data)
   {}

   void *data() const { return m_data; }
   void data( void *dt )  { m_data = dt; }
};

class DllLoader_base
{
public:
   DllLoader_base() {}
   virtual ~DllLoader_base() {}
   
   virtual bool open( const String &dll_name ) = 0;
   virtual bool close() = 0;
   virtual DllFunc getSymbol( const String &sym_name ) const = 0;
   virtual void getErrorDescription( String &descr ) const = 0;

   static bool isDllMark( char ch1, char ch2 ) { return false; }
   static const char *dllExt() { return ""; }


};

}

#endif
/* end of flc_dll.h */
