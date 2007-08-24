/*
   FALCON - The Falcon Programming Language.
   FILE: dll_mac.h
   $Id: dll_mac.h,v 1.1.1.1 2006/10/08 15:05:39 gian Exp $

   libdl Dynamic Link support
   -------------------------------------------------------------------
   Author: Guerra Francesco
   Begin: sab mag 06 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2006: the FALCON developers (see list in AUTHORS file)

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

   ~DllLoader_Mac();

   bool open( const String &dll_name );
   bool close();
   virtual void getErrorDescription( String &descr ) const;

   void assign( DllLoader_Mac &other );

   DllFunc getSymbol( const String &sym_name ) const ;
   static bool isDllMark( unsigned char ch1, unsigned char ch2 );
   static const char *dllExt() { return ".dylib"; };
};

typedef DllLoader_Mac DllLoader;

}

#endif

/* end of flc_dll_dl.h */
