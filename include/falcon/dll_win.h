/*
   FALCON - The Falcon Programming Language.
   FILE: flc_dll_win.h
   $Id: dll_win.h,v 1.1.1.1 2006/10/08 15:05:39 gian Exp $

   Windows specific class for Dynamic load system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

   ~DllLoader_win();

   bool open( const String &dll_name );
   bool close();
   virtual void getErrorDescription( String &descr ) const;

   void assign( DllLoader_win &other );

   DllFunc getSymbol( const char *sym_name ) const ;
   static bool isDllMark( char ch1, char ch2 );
   static const char *dllExt() { return ".dll"; };
};

typedef DllLoader_win DllLoader;

}

#endif

/* end of flc_dll_win.h */
