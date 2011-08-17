/*
   FALCON - The Falcon Programming Language.
   FILE: flc_dll_dl.h

   libdl Dynamic Link support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom set 12 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

   virtual ~DllLoader_dl();

   bool open( const String &dll_name );
   bool close();
   virtual void getErrorDescription( String &descr ) const;

   void assign( DllLoader_dl &other );

   DllFunc getSymbol( const String &sym_name ) const ;
   static bool isDllMark( char ch1, char ch2 );
   static const char *dllExt() { return "so"; };
};

typedef DllLoader_dl DllLoader;

}

#endif

/* end of flc_dll_dl.h */
