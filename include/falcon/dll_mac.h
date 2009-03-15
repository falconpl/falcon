/*
   FALCON - The Falcon Programming Language.
   FILE: dll_mac.h

   libdl Dynamic Link support
   -------------------------------------------------------------------
   Author: Guerra Francesco
   Begin: sab mag 06 2006

   -------------------------------------------------------------------
   (C) Copyright 2006: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   This file contains the multiplatform DLL incarnation for those
   systems implementing lib DL (dlopen/dlsym/dlclose).
*/

/**
	CONST
*/

#ifndef flc_DLL_DL_H
#define flc_DLL_DL_H
#include <falcon/dll_base.h>
#include <dlfcn.h>

namespace Falcon {

class DllLoader_Mac
{
   void *m_module;

public:
   DllLoader_Mac():
      m_module( 0 )
   {}

   virtual ~DllLoader_Mac();

   bool open( const String &dll_name );
   bool close();
   virtual void getErrorDescription( String &descr ) const;

   void assign( DllLoader_Mac &other );

   DllFunc getSymbol( const String &sym_name ) const ;
   static bool isDllMark( unsigned char ch1, unsigned char ch2 );
   static const char *dllExt() { return "dylib"; };
};

typedef DllLoader_Mac DllLoader;

}

#endif

/* end of flc_dll_dl.h */
