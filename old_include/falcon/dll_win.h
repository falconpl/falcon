/*
   FALCON - The Falcon Programming Language.
   FILE: flc_dll_win.h

   Windows specific class for Dynamic load system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_DLL_WIN_H
#define flc_DLL_WIN_H

#include <falcon/dll_base.h>
#include <windows.h>

namespace Falcon
{

class FALCON_DYN_CLASS DllLoader_win
{
   HMODULE m_module;
   DWORD m_error;

public:
   DllLoader_win():
      //DllLoader_base(),
      m_module( NULL )
   {}

   virtual ~DllLoader_win();

   bool open( const String &dll_name );
   bool close();
   virtual void getErrorDescription( String &descr ) const;

   void assign( DllLoader_win &other );

   DllFunc getSymbol( const char *sym_name ) const ;
   static bool isDllMark( char ch1, char ch2 );
   static const char *dllExt() { return "dll"; };
};

typedef DllLoader_win DllLoader;

}

#endif

/* end of flc_dll_win.h */
