/*
   FALCON - The Falcon Programming Language.
   FILE: flc_dll_dl.h
   $Id: dll_dl.h,v 1.1.1.1 2006/10/08 15:05:38 gian Exp $

   libdl Dynamic Link support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom set 12 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   This file contains the multiplatform DLL incarnation for those
   systems implementing lib DL (dlopen/dlsym/dlclose).
*/

#ifndef flc_DLL_DL_H
#define flc_DLL_DL_H
#include <falcon/dll_base.h>
#include <dlfcn.h>

namespace Falcon {

class DllLoader_dl
{
   void *m_module;

public:
   DllLoader_dl():
      m_module( 0 )
   {}

   ~DllLoader_dl();

   bool open( const String &dll_name );
   bool close();
   virtual void getErrorDescription( String &descr ) const;

   void assign( DllLoader_dl &other );

   DllFunc getSymbol( const String &sym_name ) const ;
   static bool isDllMark( char ch1, char ch2 );
   static const char *dllExt() { return ".so"; };
};

typedef DllLoader_dl DllLoader;

}

#endif

/* end of flc_dll_dl.h */
